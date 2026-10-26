// User-facing review races and persistence failures, without external packages.
const assert=require('node:assert/strict');
const fs=require('node:fs'),vm=require('node:vm');
const html=fs.readFileSync('src/web/ball_testing.html','utf8');
const htmlIds=[...html.matchAll(/\bid="([^"]+)"/g)].map(match=>match[1]);
assert.equal(new Set(htmlIds).size,htmlIds.length,'Duplicate UI element IDs');
for(const match of html.matchAll(/\$\('([^']+)'\)/g))assert.ok(htmlIds.includes(match[1]),`Missing element ${match[1]}`);
const source=html.match(/<script>([\s\S]*?)<\/script>/)[1];
new vm.Script(source);
const elements=new Map(),images=[];
const drawingCalls=[];
const drawing=new Proxy({},{get:(_target,name)=>(...args)=>drawingCalls.push({name,args}),set:()=>true});
function element(id){return {id,value:'',hidden:false,disabled:false,textContent:'',children:[],width:1920,height:1080,
  replaceChildren(){this.children=[]},appendChild(child){this.children.push(child)},getContext(){return drawing},
  getBoundingClientRect(){return {left:20,top:30,width:960,height:540}}};}
class ImageStub{constructor(){this.naturalWidth=1920;this.naturalHeight=1080;images.push(this)}set src(value){this.url=value}}
let requests=[],fetcher;
const sandbox={document:{getElementById(id){if(!elements.has(id))elements.set(id,element(id));return elements.get(id)},createElement:element,addEventListener(){}},
  window:{addEventListener(){}},Image:ImageStub,fetch:async(url,options)=>{requests.push({url,options});return fetcher(url,options)},
  history:{replaceState(){}},location:{search:''},URLSearchParams,setTimeout:()=>1,clearTimeout(){},console,Math,Number,JSON,Map,Set};
vm.createContext(sandbox);vm.runInContext(source.replace(/\s*init\(\);\s*$/,''),sandbox);
const run=code=>vm.runInContext(code,sandbox);
const response=(data,ok=true)=>({ok,json:async()=>data});
function seed(){requests=[];run(`state.session='abc';state.index=0;state.busy=false;state.loading=false;state.draft=null;state.cursor=null;state.aimDirty=false;state.aimOrigin=null;
$('magnifier').width=480;$('magnifier').height=360;$('zoom').value='5';$('notes').value='';$('afterLabel').value='next';
state.image={naturalWidth:1920,naturalHeight:1080};state.data={metadata:{fps:30,status:'completed',split:'train',model_version:'ball_v1'},annotations:{revision:4,labels:[]},records:[
{frame:100,court_ok:true,classical:[500,200],neural:[510,205],classical_score:.8,neural_score:.97},
{frame:101,court_ok:true,classical:null,neural:null,classical_score:0,neural_score:.2}]};`);}
(async()=>{
  seed();let finish;fetcher=()=>new Promise(resolve=>finish=resolve);
  const saving=run(`saveAnnotation({frame:100,visibility:'visible',x:510,y:205,notes:'ball'})`);
  assert.equal(elements.get('next').disabled,true);
  assert.equal(JSON.parse(requests[0].options.body).expected_revision,4);
  await run('loadFrame(1)');assert.equal(run('state.index'),0);
  finish(response({revision:5,reviewed:1,label:{frame:100,visibility:'visible',x:510,y:205,notes:'ball'}}));
  await saving;assert.equal(run('state.data.annotations.revision'),5);
  assert.equal(run('state.draft'),null);assert.match(elements.get('status').textContent,/1\/2 frames reviewed/);

  seed();fetcher=async()=>response({detail:'Disk is full'},false);
  await run(`saveAnnotation({frame:100,visibility:'visible',x:510,y:205,notes:''})`);
  assert.equal(run('state.data.annotations.revision'),4);
  assert.equal(run('state.data.annotations.labels.length'),0);
  assert.ok(run('state.draft'));assert.match(elements.get('error').textContent,/still unsaved/);
  await run('loadFrame(1)');assert.equal(run('state.index'),0);
  fetcher=async()=>response({revision:5,label:{frame:100,visibility:'visible',x:510,y:205,notes:''}});
  await elements.get('retry').onclick();assert.equal(run('state.draft'),null);

  seed();const before=images.length;await run('loadFrame(0)');await run('loadFrame(1)');
  const old=images[before],current=images[before+1];current.onload();old.onload();
  assert.equal(run('state.image'),current);assert.match(elements.get('frameInfo').textContent,/Frame 101/);

  seed();fetcher=async()=>response({revision:5,label:{frame:100,visibility:'occluded',x:null,y:null,notes:''}});
  await run(`saveAnnotation({frame:100,visibility:'occluded',x:null,y:null,notes:''})`);
  const hidden=JSON.parse(requests[0].options.body);assert.equal(hidden.x,null);assert.equal(hidden.y,null);
  assert.equal(run('state.data.annotations.labels[0].visibility'),'occluded');
  assert.equal(elements.get('labelStatus').textContent,'Saved: Hidden / blocked');

  seed();fetcher=async(_url,options)=>response({revision:5,label:JSON.parse(options.body)});
  await run("visibility('absent')");
  assert.equal(run('state.data.annotations.labels[0].visibility'),'absent');
  assert.equal(elements.get('labelStatus').textContent,'Saved: Outside frame');
  assert.equal(JSON.parse(requests[0].options.body).x,null);

  seed();fetcher=async()=>response({detail:'Changed in another window'},false);
  await run(`saveAnnotation({frame:100,visibility:'visible',x:510,y:205,notes:''})`);
  elements.get('discard').onclick();assert.equal(run('state.draft'),null);
  assert.equal(run('state.data.annotations.revision'),4);

  seed();const point=run('coordinate({clientX:99999,clientY:-100})');
  assert.equal(point.x,1919);assert.equal(point.y,0);

  // Carry the previous reviewed position, but do not fabricate a label.
  seed();run(`state.data.annotations.labels=[{frame:100,visibility:'visible',x:510.5,y:205.25,notes:''}];state.cursor={x:510.5,y:205.25}`);
  await run('loadFrame(1)');images.at(-1).onload();
  assert.equal(run('state.cursor.x'),510.5);assert.equal(run('state.cursor.y'),205.25);
  assert.equal(run('label()'),undefined);assert.equal(requests.length,0);
  assert.match(elements.get('aimStatus').textContent,/Preview only/);
  assert.equal(run('state.data.annotations.labels.length'),1);

  // Arrows adjust source pixels; Shift is a larger step. Form controls retain normal keys.
  const key=(name,extras={})=>{sandbox.keyEvent={key:name,target:{tagName:'DIV'},preventDefault(){},...extras};return run('reviewKey(keyEvent)')};
  await key('ArrowRight');await key('ArrowUp',{shiftKey:true});
  assert.equal(run('state.cursor.x'),511.5);assert.equal(run('state.cursor.y'),200.25);
  assert.equal(run('state.index'),1);assert.equal(requests.length,0);
  await key('ArrowDown',{target:{tagName:'TEXTAREA'}});assert.equal(run('state.cursor.y'),200.25);
  await key('ArrowDown',{ctrlKey:true});assert.equal(run('state.cursor.y'),200.25);
  assert.equal(run('samePoint({x:0,y:0},{x:null,y:null})'),false);

  // Confirm + next sends the adjusted coordinates exactly once and carries them forward.
  seed();run('state.cursor={x:500,y:200}');await key('ArrowRight');
  fetcher=async(_url,options)=>{const saved=JSON.parse(options.body);return response({revision:5,label:saved})};
  await key('Enter');assert.equal(requests.length,1);
  assert.equal(JSON.parse(requests[0].options.body).x,501);
  assert.equal(JSON.parse(requests[0].options.body).visibility,'visible');
  assert.equal(run('state.index'),1);images.at(-1).onload();
  assert.equal(run('state.cursor.x'),501);assert.equal(run('label()'),undefined);
  assert.equal(run('state.data.annotations.labels.length'),1);
  await key('Enter',{repeat:true});assert.equal(requests.length,1);

  // P confirms partial at the adjusted point; failed saves never advance.
  seed();run('state.cursor={x:400,y:220}');
  fetcher=async(_url,options)=>response({revision:5,label:JSON.parse(options.body)});
  await key('p');assert.equal(JSON.parse(requests[0].options.body).visibility,'partial');
  assert.equal(run('state.index'),1);
  seed();run('state.cursor={x:400,y:220}');fetcher=async()=>response({detail:'Disk full'},false);
  await key('Enter');assert.equal(run('state.index'),0);assert.ok(run('state.draft'));

  // Existing references on a destination frame override carried previews.
  seed();run(`state.cursor={x:400,y:220};state.data.annotations.labels=[{frame:101,visibility:'partial',x:600,y:330,notes:''}]`);
  await run('loadFrame(1)');images.at(-1).onload();
  assert.equal(run('state.cursor.x'),600);assert.equal(run('label().visibility'),'partial');
  assert.equal(requests.length,0);

  // Red dot remains exactly at the crop center even against a source-image edge.
  seed();run('state.cursor={x:0,y:0}');drawingCalls.length=0;run('drawMagnifier()');
  const crop=drawingCalls.find(item=>item.name==='drawImage');
  assert.equal(crop.args[1],0);assert.equal(crop.args[2],0);
  assert.equal(crop.args[5],240);assert.equal(crop.args[6],180);
  const dot=drawingCalls.find(item=>item.name==='arc');
  assert.equal(dot.args[0],240);assert.equal(dot.args[1],180);
  assert.equal(dot.args[2],2.2);

  // Physical numpad codes work with Num Lock off; all labels save + advance.
  for(const [code,keyName,visibility] of [
    ['Numpad7','Home','visible'],['Numpad9','PageUp','partial'],
    ['Numpad1','End','occluded'],['Numpad0','Insert','absent'],
    ['Numpad3','PageDown','uncertain'],['NumpadEnter','Enter','visible']]){
    seed();if(['visible','partial'].includes(visibility))run('state.cursor={x:400,y:220}');
    fetcher=async(_url,options)=>response({revision:5,label:JSON.parse(options.body)});
    await key(keyName,{code});
    assert.equal(requests.length,1,code);
    const payload=JSON.parse(requests[0].options.body);
    assert.equal(payload.visibility,visibility,code);
    assert.equal(payload.x,['visible','partial'].includes(visibility)?400:null);
    assert.equal(run('state.index'),1,code);
  }

  // Num Lock off does not turn a single nudge into both a pad and arrow action.
  seed();run('state.cursor={x:400,y:220}');
  await key('ArrowDown',{code:'Numpad2'});assert.equal(run('state.cursor.y'),221);
  await key('ArrowLeft',{code:'Numpad4',shiftKey:true});assert.equal(run('state.cursor.x'),395);
  await key('7',{code:'Digit7'});assert.equal(requests.length,0);
  await key('Insert',{code:'Numpad0',repeat:true});assert.equal(requests.length,0);
  await key('Insert',{code:'Numpad0',target:{tagName:'INPUT'}});assert.equal(requests.length,0);

  // Label buttons and numpad obey Stay; later + advances without another save.
  seed();run("$('afterLabel').value='stay'");
  fetcher=async(_url,options)=>response({revision:5,label:JSON.parse(options.body)});
  await elements.get('absent').onclick();assert.equal(run('state.index'),0);
  assert.equal(requests.length,1);
  await key('+',{code:'NumpadAdd'});assert.equal(run('state.index'),1);
  assert.equal(requests.length,1);images.at(-1).onload();
  await key('-',{code:'NumpadSubtract'});assert.equal(run('state.index'),0);

  // Save + next unreviewed skips existing labels, and never labels the destination.
  seed();run(`$('afterLabel').value='unreviewed';state.cursor={x:400,y:220};
    state.data.records.push({...state.data.records[1],frame:102});
    state.data.annotations.labels=[{frame:101,visibility:'absent',x:null,y:null,notes:''}];`);
  fetcher=async(_url,options)=>response({revision:5,label:JSON.parse(options.body)});
  await key('Insert',{code:'Numpad0'});
  assert.equal(run('state.index'),2);assert.equal(run('state.data.annotations.labels.length'),2);
  assert.equal(JSON.parse(requests[0].options.body).frame,100);
  images.at(-1).onload();assert.equal(run('label()'),undefined);

  // Decimal is next unreviewed even when the key's text is Delete.
  seed();run(`state.data.records.push({...state.data.records[1],frame:102});
    state.data.annotations.labels=[{frame:101,visibility:'absent',x:null,y:null,notes:''}];`);
  await key('Delete',{code:'NumpadDecimal'});assert.equal(run('state.index'),2);
  assert.equal(requests.length,0);

  seed();fetcher=async()=>response({detail:'Disk full'},false);
  await key('Insert',{code:'Numpad0'});assert.equal(run('state.index'),0);
  await key('Delete',{code:'NumpadDecimal'});assert.equal(run('state.index'),0);
  assert.ok(run('state.draft'));

  // Task 2 / Finding B: Prefill and unconfirmed suggestions do not inflate reviewed counts, accuracy, or completion.
  seed();
  run(`state.data.annotations.labels=[{frame:100,visibility:'visible',x:510,y:205,origin:'prefilled',confirmed:false,notes:'prefill:neural'}];
  showMetrics();`);
  assert.match(elements.get('status').textContent,/0\/2 frames reviewed/);
  assert.match(elements.get('status').textContent,/1 unconfirmed suggestion/);
  assert.match(elements.get('learningCounts').textContent,/0 visible centers.*1 suggestions awaiting confirmation/);
  assert.match(elements.get('metrics').children[1].textContent,/0 correct.*of 0 visible references/);
  assert.equal(elements.get('progress').value,0);

  // nextUnreviewed navigates to unconfirmed suggestions so reviewer inspects them
  run('state.index=1');
  await run('nextUnreviewed()');
  images.at(-1).onload();
  assert.equal(run('state.index'),0);

  // Confirming the suggestion makes it eligible, updates verified metrics, and preserves provenance
  fetcher=async(_url,options)=>{
    const body=JSON.parse(options.body);
    assert.deepEqual(body.frames,[100]);
    return response({revision:5,confirmed_count:1,reviewed:1});
  };
  await run('confirmCurrentFrame()');
  assert.equal(run('state.data.annotations.labels[0].confirmed'),true);
  assert.equal(run('state.data.annotations.labels[0].origin'),'prefilled');
  assert.match(elements.get('status').textContent,/1\/2 frames reviewed/);
  assert.match(elements.get('metrics').children[1].textContent,/1 correct.*of 1 visible references/);
  assert.equal(elements.get('progress').value,1);

  // Legacy prefilled/interpolated notes without origin are recognized as unconfirmed suggestions
  seed();
  run(`state.data.annotations.labels=[{frame:100,visibility:'visible',x:510,y:205,notes:'prefill:classical'}];
  showMetrics();`);
  assert.match(elements.get('status').textContent,/0\/2 frames reviewed/);
  assert.match(elements.get('status').textContent,/1 unconfirmed suggestion/);
  assert.equal(run('isConfirmed(state.data.annotations.labels[0])'),false);

  console.log('PASS: numpad labels/navigation with Num Lock off, advance modes, single-pixel nudges, repeats/edit-field guards, autosave failures, preview isolation, frame races, magnifier centering, suggestion confirmation isolation, and legacy suggestion recognition.');
})().catch(error=>{console.error(error);process.exitCode=1});
