/* 몸으로 체육대회 — 동작 인식 엔진 어댑터
   MediaPipe Tasks PoseLandmarker(여러 명 동시 인식)를 옛 Pose 솔루션과 같은 모양으로 감쌈.
   - 한 화면에 여러 명이 보여도 '인식 구역(window.__zone) 안에서 가운데에 가장 가까운 사람'을 따라감
   - 한 번 고른 사람을 잘 안 놓치게(끈적임) 처리
   - 새 엔진을 못 쓰는 기기는 옛 엔진(한 명 인식)으로 자동 전환 */
const HERE=new URL('.',import.meta.url).href;
const LEGACY='https://cdn.jsdelivr.net/npm/@mediapipe/pose@0.5.1675469404/';
window.__poseInfo={engine:'loading',people:0,picked:-1};

function loadScript(src){return new Promise((ok,no)=>{const s=document.createElement('script');s.src=src;s.crossOrigin='anonymous';s.onload=ok;s.onerror=()=>no(new Error('load '+src));document.head.appendChild(s)})}

class PoseShim{
  constructor(opts){this.opts=opts||{};this.cb=null;this.lm=null;this.leg=null;this.lastTs=0;this.prev=null;this.sel={};this.numPoses=3}
  setOptions(o){this.sel=o||{}}
  onResults(f){this.cb=f;if(this.leg)this.leg.onResults(f)}
  async initialize(){
    try{
      const mod=await import(HERE+'vision_bundle.mjs');
      const files=await mod.FilesetResolver.forVisionTasks(HERE+'wasm');
      const make=async delegate=>mod.PoseLandmarker.createFromOptions(files,{
        baseOptions:{modelAssetPath:HERE+'pose_landmarker_lite.task',delegate},
        runningMode:'VIDEO',numPoses:this.numPoses,
        minPoseDetectionConfidence:.5,minPosePresenceConfidence:.5,minTrackingConfidence:.5});
      try{this.lm=await make('GPU')}catch(e){this.lm=await make('CPU')}
      window.__poseInfo.engine='tasks';
    }catch(e){
      console.warn('새 인식 엔진을 쓰지 못해 옛 엔진으로 전환:',e);
      await loadScript(LEGACY+'pose.js');
      const L=window.Pose; // 이제 옛 엔진 클래스
      this.leg=new L({locateFile:f=>LEGACY+f});
      this.leg.setOptions({modelComplexity:0,smoothLandmarks:true,enableSegmentation:false,minDetectionConfidence:.5,minTrackingConfidence:.5});
      if(this.cb)this.leg.onResults(this.cb);
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
  async send({image}){
    if(this.leg)return this.leg.send({image});
    if(!this.lm||!this.cb)return;
    let ts=performance.now();if(ts<=this.lastTs)ts=this.lastTs+1;this.lastTs=ts;
    let res;
    try{res=this.lm.detectForVideo(image,ts)}catch(e){return}
    const poses=(res&&res.landmarks)||[];
    const chosen=this.pick(poses);
    this.cb({poseLandmarks:chosen?chosen.map(q=>({x:q.x,y:q.y,z:q.z,visibility:q.visibility??1})):null,allPoses:poses,image});
  }
  close(){try{this.lm&&this.lm.close()}catch(e){}}
}
window.Pose=PoseShim;
window.__poseReady=Promise.resolve();
