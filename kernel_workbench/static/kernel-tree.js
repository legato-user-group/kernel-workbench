const natural = new Intl.Collator('en', {numeric:true});

export function modulePrefixes(module) {
  const parts=String(module || 'unattributed').split('.'), prefixes=[];
  for(let i=0;i<parts.length;i++){
    if(/^\d+$/.test(parts[i+1] || '')) i++;
    prefixes.push(parts.slice(0,i+1).join('.'));
  }
  return prefixes;
}

export function kernelTree(rows, model) {
  const labels=new Map(), order=new Map();let position=0;
  function index(nodes){for(const n of nodes||[]){if(n.id){labels.set(n.id,n.label);order.set(n.id,position++)}index(n.children)}}
  index(model.children);
  const root={id:'$model',label:model.name,children:[],entries:[]}, nodes=new Map([['$model',root]]);
  rows.forEach((row,index)=>{
    let parent=root;
    for(const id of modulePrefixes(row.module)){
      if(!nodes.has(id)){
        const label=id==='unattributed'?'소속 미확인 · unattributed':labels.get(id)||id.split('.').at(-1);
        const node={id,label,children:[],entries:[]};nodes.set(id,node);parent.children.push(node);
      }
      parent=nodes.get(id);
    }
    parent.entries.push({row,index});
  });
  function rank(node){return Math.min(order.get(node.id)??Infinity,...node.children.map(rank))}
  function sort(node){
    node.children.sort((a,b)=>a.id==='unattributed'?1:b.id==='unattributed'?-1:
      rank(a)-rank(b)||natural.compare(a.id,b.id));
    node.entries.sort((a,b)=>({prefill:0,decode:1}[a.row.phase]??2)-({prefill:0,decode:1}[b.row.phase]??2)||a.index-b.index);
    node.children.forEach(sort);
  }
  sort(root);return root;
}
