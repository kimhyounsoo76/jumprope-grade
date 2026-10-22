/* 몸으로 체육대회 — 동작 인식 엔진 어댑터
   MediaPipe Tasks PoseLandmarker(여러 명 동시 인식)를 옛 Pose 솔루션과 같은 모양으로 감쌈.
   - 한 화면에 여러 명이 보여도 '인식 구역(window.__zone) 안에서 가운데에 가장 가까운 사람'을 따라감
   - 한 번 고른 사람을 잘 안 놓치게(끈적임) 처리
   - 새 엔진을 못 쓰는 기기는 옛 엔진(한 명 인식)으로 자동 전환 */
const HERE=new URL('.',import.meta.url).href;
const LEGACY='https://cdn.jsdelivr.net/npm/@mediapipe/pose@0.5.1675469404/';
window.__poseInfo={engine:'loading',people:0,picked:-1};
let LegacyPose=null;

function loadScript(src){return new Promise((ok,no)=>{const s=document.createElement('script');s.src=src;s.crossOrigin='anonymous';s.onload=ok;s.onerror=()=>no(new Error('load '+src));document.head.appendChild(s)})}

class PoseShim{
  constructor(opts){this.opts=opts||{};this.cb=null;this.lm=null;this.leg=null;this.lastTs=0;this.prev=null;this.sel={};this.numPoses=2;this.prevPoses=[];this.busy=false;this.cost=0}
  setOptions(o){this.sel=o||{}}
  onResults(f){this.cb=f;if(this.leg)this.leg.onResults(r=>this.legacyResult(r))}
  // 옛 엔진(한 명): 인식 구역이 전체보다 좁으면 가운데만 잘라 보내고, 좌표를 전체 화면 기준으로 되돌림
  legacyResult(r){
    if(this.cb){const z=this.cropZ||1;if(r.poseLandmarks&&z<1){const x0=(1-z)/2;for(const q of r.poseLandmarks)q.x=x0+q.x*z}
      window.__poseInfo.people=r.poseLandmarks?1:0;window.__poseInfo.picked=r.poseLandmarks?0:-1;this.cb(r)}
  }
  // 설정: pegames.v1.engine = 'tasks'(새, 여러 명) | 'legacy'(옛, 한 명·가벼움). 기본은 새 엔진
  pref(){try{return (JSON.parse(localStorage.getItem('pegames.v1')||'{}').engine)||'auto'}catch(e){return 'auto'}}
  async initialize(){
    try{
      // 기본은 검증된 옛 엔진(한 명·가벼움). 새 엔진은 설정에서 '여러 명 인식'을 고르거나, 두 명이 필요한 게임(피클볼)일 때만
      const want=this.pref();
      if(want==='legacy'||(want!=='tasks'&&!window.__needMulti))throw new Error('옛 엔진 사용');
      const mod=await import(HERE+'vision_bundle.mjs');
      const files=await mod.FilesetResolver.forVisionTasks(HERE+'wasm');
      const make=async delegate=>mod.PoseLandmarker.createFromOptions(files,{
        baseOptions:{modelAssetPath:HERE+'pose_landmarker_lite.task',delegate},
        runningMode:'VIDEO',numPoses:this.numPoses,
        minPoseDetectionConfidence:.5,minPosePresenceConfidence:.5,minTrackingConfidence:.5});
      try{this.lm=await make('GPU')}catch(e){this.lm=await make('CPU')}
      window.__poseInfo.engine='tasks';
    }catch(e){
      if(!/옛 엔진 사용/.test(String(e&&e.message)))console.warn('새 인식 엔진을 쓰지 못해 옛 엔진으로 전환:',e);
      if(!LegacyPose){await loadScript(LEGACY+'pose.js');LegacyPose=window.Pose;window.Pose=PoseShim} // 옛 엔진 클래스는 따로 두고, 전역 Pose는 계속 어댑터
      const L=LegacyPose;
      this.leg=new L({locateFile:f=>LEGACY+f});
      this.leg.setOptions({modelComplexity:0,smoothLandmarks:true,enableSegmentation:false,minDetectionConfidence:.5,minTrackingConfidence:.5});
      this.leg.onResults(r=>this.legacyResult(r));
      await this.leg.initialize();
      window.__poseInfo.engine='legacy';
    }
  }
  // 인식 구역 안에서 가운데에 가장 가까운 사람 고르기 (직전에 고른 사람에게 가산점)
  pick(poses){
    const z=Math.min(1,Math.max(.3,+window.__zone||1)),lo=(1-z)/2,hi=(1+z)/2;
    let best=null,bestScore=1e9,bestIdx=-1;
    poses.forEach((p,idx)=>{
      const a=p[11],b=p[12],c=p[23],d=p[24];
      if(!a||!b||!c||!d)return;
      const vis=Math.min(a.visibility??1,b.visibility??1);
      if(vis<.3)return;
      const hx=(c.x+d.x)/2,hy=(c.y+d.y)/2,cx=(a.x+b.x+c.x+d.x)/4;
      if(cx<lo||cx>hi)return; // 구역 밖 사람은 무시
      let score=Math.abs(cx-.5);
      if(this.prev){const dd=Math.hypot(hx-this.prev.x,hy-this.prev.y);if(dd<.12)score-=.2} // 같은 사람 계속
      if(score<bestScore){bestScore=score;best=p;bestIdx=idx}
    });
    if(best){const c=best[23],d=best[24];this.prev={x:(c.x+d.x)/2,y:(c.y+d.y)/2}}
    else if(this.prev&&(performance.now()-(this.prevAt||0))>1500)this.prev=null;
    if(best)this.prevAt=performance.now();
    window.__poseInfo.people=poses.length;window.__poseInfo.picked=bestIdx;
    return best;
  }
  // 같은 사람이 두 번 잡히는 일(겹친 인식) 정리: 엉덩이·어깨 중심이 거의 같은 자세는 더 또렷한 쪽 하나만 남김
  dedup(poses){
    const info=poses.map(p=>({p,hx:(p[23].x+p[24].x)/2,hy:(p[23].y+p[24].y)/2,sx:(p[11].x+p[12].x)/2,sy:(p[11].y+p[12].y)/2,
      vis:p.reduce((a,q)=>a+(q.visibility??1),0)/p.length}));
    info.sort((a,b)=>b.vis-a.vis);
    const keep=[];
    for(const c of info){
      const dup=keep.some(k=>Math.hypot(k.hx-c.hx,k.hy-c.hy)<.08&&Math.hypot(k.sx-c.sx,k.sy-c.sy)<.1);
      if(!dup)keep.push(c);
    }
    return keep.map(k=>k.p);
  }
  smooth(poses){
    const out=poses.map(p=>{
      const hx=(p[23].x+p[24].x)/2,hy=(p[23].y+p[24].y)/2;
      let best=null,bd=.15;
      for(const q of this.prevPoses){const d=Math.hypot(q.hx-hx,q.hy-hy);if(d<bd){bd=d;best=q}}
      const lm=p.map((a,i)=>{
        if(!best)return{x:a.x,y:a.y,z:a.z,visibility:a.visibility??1};
        const b=best.lm[i],d=Math.hypot(a.x-b.x,a.y-b.y);
        const k=Math.min(1,.25+d*10); // 1cm 떨림은 1/4만 반영, 10%쯤 움직이면 그대로
        return{x:b.x+(a.x-b.x)*k,y:b.y+(a.y-b.y)*k,z:a.z,visibility:a.visibility??1};
      });
      return{lm,hx,hy};
    });
    this.prevPoses=out;
    return out.map(o=>o.lm);
  }
  async send({image}){
    if(this.leg){
      const z=Math.min(1,Math.max(.3,+window.__zone||1));this.cropZ=z;
      if(z>=1||!image.videoWidth)return this.leg.send({image});
      const vw=image.videoWidth,vh=image.videoHeight,cw=Math.round(vw*z),cx=Math.round((vw-cw)/2);
      if(!this.cropCv){this.cropCv=document.createElement('canvas')}
      if(this.cropCv.width!==cw||this.cropCv.height!==vh){this.cropCv.width=cw;this.cropCv.height=vh;this.cropCtx=this.cropCv.getContext('2d')}
      this.cropCtx.drawImage(image,cx,0,cw,vh,0,0,cw,vh);
      return this.leg.send({image:this.cropCv});
    }
    if(!this.lm||!this.cb)return;
    let ts=performance.now();if(ts<=this.lastTs)ts=this.lastTs+1;this.lastTs=ts;
    // 느린 기기: 한 번 인식에 40ms 넘게 걸리면 그림이 끊기지 않게 한 프레임 걸러 인식
    if(this.cost>40&&this.skip){this.skip=false;return}this.skip=true;
    let res;const t0=performance.now();
    try{res=this.lm.detectForVideo(image,ts)}catch(e){return}
    this.cost=this.cost*.8+(performance.now()-t0)*.2;window.__poseInfo.ms=Math.round(this.cost);
    const poses=this.smooth(this.dedup((res&&res.landmarks)||[]));
    const chosen=this.pick(poses);
    this.cb({poseLandmarks:chosen?chosen.map(q=>({x:q.x,y:q.y,z:q.z,visibility:q.visibility??1})):null,allPoses:poses,image});
  }
  close(){try{this.lm&&this.lm.close()}catch(e){}}
}
window.Pose=PoseShim;
window.__poseReady=Promise.resolve();
