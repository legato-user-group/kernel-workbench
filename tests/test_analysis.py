import json
import sqlite3
import pytest
from kernel_workbench.analysis import bounds,hardware,read_nsys,read_ncu,analyze,counters


def test_bounds_and_units():
    hw=hardware('A100-SXM4-40GB')
    assert bounds({'bf16_tensor':312e9},1555e6,hw)['lower_ms']==pytest.approx(1)
    assert bounds({'fp32':19.5e9},1,hw)['bound']=='compute'
    assert bounds({'fp32':1},1555e6,hw)['bound']=='memory'
    assert bounds({'fp32':19.5e9,'bf16_tensor':624e9},1,hw)['compute_ms']==pytest.approx(2)
    assert bounds(None,0,hw)['bound']=='unknown'
    assert bounds({'fp32':1},None,hw)['lower_ms'] is None
    assert bounds({'fp32':0},0,hw)['lower_ms']==0
    assert bounds({'fp32':1},1,hardware('MIG A100'))['lower_ms'] is None


def test_t4_roofline_uses_fp16_and_has_no_bf16_ceiling():
    hw = hardware('Tesla T4')
    assert hw['bandwidth'] == 300e9 and hw['peaks']['fp16_tensor'] == 65e12
    assert 'bf16_tensor' not in hw['peaks']
    assert bounds({'fp16_tensor': 65e9}, 1, hw)['compute_ms'] == pytest.approx(1)
    assert bounds({'bf16_tensor': 1}, 1, hw)['bound'] == 'unknown'


def test_launch_side_attribution(tmp_path):
    path=tmp_path/'trace.sqlite'; db=sqlite3.connect(path)
    db.executescript('''CREATE TABLE StringIds(id INTEGER,value TEXT);
    INSERT INTO StringIds VALUES(1,'gemm');
    CREATE TABLE NVTX_EVENTS(start INTEGER,end INTEGER,globalTid INTEGER,text TEXT);
    INSERT INTO NVTX_EVENTS VALUES(10,50,1,'inference'),(11,49,1,'phase:decode'),(12,48,1,'step:3'),(13,47,1,'module:layer.0.qkv');
    CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER,end INTEGER,globalTid INTEGER,correlationId INTEGER);
    INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(20,21,1,7);
    CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,correlationId INTEGER,shortName INTEGER);
    INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(100,200,7,1);''')
    db.close()
    k=read_nsys(path)[0]
    assert k['module']=='layer.0.qkv' and k['phase']=='decode' and k['step']==3
    assert k['duration_ms']==.0001


def test_csv_and_fma(tmp_path):
    p=tmp_path/'raw.csv'
    p.write_text('banner\n"ID","Kernel Name","Metric Name","Metric Unit","Metric Value"\n"0","gemm","dram__bytes_read.sum","byte","1,024"\n"0","gemm","dram__bytes_write.sum","byte","512"\n')
    rows=read_ncu(p); assert rows[0]['metrics']['dram__bytes_read.sum']==1024
    f,d=counters(rows[0]['metrics']);assert f is None and d==1536
    metrics={f'smsp__sass_thread_inst_executed_op_f{o}_pred_on.sum':1 for o in ('add','mul','fma')}
    metrics.update({f'sm__ops_path_tensor_src_{k}_dst_fp32.sum':0 for k in ('bf16','fp16','tf32')})
    assert counters(metrics)[0]['fp32']==4


def test_mismatch_never_fabricates_kernel_counters():
    meta=json.load(open('kernel_workbench/static/example.json'))
    source={'gpu':{'name':meta['hardware']['name']},'model':meta['model'],'config':meta['config'],
            'total_ms':9.4,'phase_ms':{'prefill':1.8,'decode':7.6},'estimates':meta['estimates']}
    k={**meta['kernels'][0],'source':'unavailable','flops':None,'bytes':None}
    result=analyze(source,[k],[{'name':'different','metrics':{}}])
    assert result['coverage']['measured']==0
    assert result['point']['source']=='estimated'
    assert result['kernels'][0]['lower_ms'] is None
    assert result['point']['tflops']==pytest.approx(result['point']['flops']/.0094/1e12)


def test_wide_ncu_export(tmp_path):
    p=tmp_path/'wide.csv'
    p.write_text('"ID","Kernel Name","Grid Size","Block Size","dram__bytes_read.sum","dram__bytes_write.sum"\n"","","","","byte","byte"\n"0","gemm","(2, 1, 1)","(32, 4, 1)","1,024","512"\n')
    row=read_ncu(p)[0]
    assert row['grid']==[2,1,1] and row['block']==[32,4,1]
    assert row['metrics']['dram__bytes_read.sum']==1024


def test_phase_rooflines_use_phase_totals_and_phase_time():
    from kernel_workbench.analysis import with_phase_points
    result = {
        'point': {'source': 'measured'}, 'config': {'output_tokens': 3},
        'timings': {'total_ms': 12, 'prefill': 4, 'decode': 8},
        'estimates': [], 'kernels': [
            {'phase': 'prefill', 'source': 'measured', 'flops': {'fp32': 4000}, 'bytes': 100},
            {'phase': 'decode', 'source': 'measured', 'flops': {'fp32': 300}, 'bytes': 50},
            {'phase': 'decode', 'source': 'measured', 'flops': {'fp32': 500}, 'bytes': 150},
        ]}
    enriched = with_phase_points(result)
    prefill, decode = [enriched['phase_points'][p] for p in ('prefill','decode')]
    assert prefill['intensity'] == 40 and decode['intensity'] == 4
    assert prefill['tflops'] == pytest.approx(4000/.004/1e12)
    assert decode['tflops'] == pytest.approx(800/.008/1e12)
    assert decode['duration_ms'] == 8 and decode['bytes'] == 200
    assert enriched['point'] == result['point']


def test_no_decode_is_explicit_even_with_sync_duration():
    from kernel_workbench.analysis import with_phase_points
    result={'point':None,'config':{'output_tokens':1},'timings':{'prefill':2,'decode':.02},'kernels':[], 'estimates':[]}
    result=with_phase_points(result)
    assert result['phase_points']['decode'] is None
    assert result['phase_point_reasons']['decode']=='not_executed'


def test_partial_phase_falls_back_without_mixing_counter_and_estimate():
    from kernel_workbench.analysis import with_phase_points
    result={'point':None,'config':{'output_tokens':2},'timings':{'prefill':1,'decode':2},
            'kernels':[{'phase':'decode','source':'measured','flops':{'fp32':9999},'bytes':9999},
                       {'phase':'decode','source':'partial','flops':None,'bytes':None}],
            'estimates':[{'phase':'decode','flops':{'fp32':20},'bytes':10}]}
    point=with_phase_points(result)['phase_points']['decode']
    assert point['source']=='estimated' and point['flops']==20 and point['bytes']==10


def test_zero_traffic_phase_not_plotted():
    from kernel_workbench.analysis import with_phase_points
    result={'point':None,'config':{'output_tokens':2},'timings':{'prefill':1,'decode':2},'estimates':[],
            'kernels':[{'phase':'decode','source':'measured','flops':{'fp32':20},'bytes':0}]}
    assert with_phase_points(result)['phase_points']['decode'] is None


def test_graph_kernel_inherits_capture_module_through_clones(tmp_path):
    path = tmp_path / 'trace.sqlite'
    db = sqlite3.connect(path)
    db.executescript('''
        CREATE TABLE StringIds(id INTEGER,value TEXT);
        INSERT INTO StringIds VALUES(1,'gemv'),(2,'copy');
        CREATE TABLE NVTX_EVENTS(start INTEGER,end INTEGER,globalTid INTEGER,text TEXT);
        INSERT INTO NVTX_EVENTS VALUES(10,40,1,'module:layer.0.mlp'),
              (11,39,1,'module:layer.0.mlp.gate_up_proj'),
              (100,150,1,'inference'),(101,149,1,'phase:decode'),(110,140,1,'step:0'),
              (111,139,1,'module:execution.decode');
        CREATE TABLE CUDA_GRAPH_NODE_EVENTS(start INTEGER,end INTEGER,globalTid INTEGER,
              graphNodeId INTEGER,originalGraphNodeId INTEGER);
        INSERT INTO CUDA_GRAPH_NODE_EVENTS VALUES(20,20,1,100,NULL),(45,45,1,101,100),(115,115,1,102,101);
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER,end INTEGER,globalTid INTEGER,correlationId INTEGER);
        INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(120,121,1,7),(20,21,1,8);
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,correlationId INTEGER,
              shortName INTEGER,graphNodeId INTEGER);
        INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(1000,2000,7,1,102),(2000,3000,7,2,NULL),
                                                        (40,90,8,1,100);
    ''')
    db.close()
    kernels = read_nsys(path)
    assert len(kernels) == 2  # Capture warmup is not counted as inference.
    assert kernels[0]['module'] == 'layer.0.mlp.gate_up_proj'
    assert kernels[0]['module_source'] == 'graph_creation_nvtx'
    assert kernels[0]['phase'] == 'decode' and kernels[0]['step'] == 0
    assert kernels[1]['module'] == 'execution.decode'
    assert kernels[1]['module_source'] == 'launch_nvtx'
