// Run with: node tests/test_calibration_ui.cjs
// Exercise the real inline UI logic without a browser or third-party packages.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const html = fs.readFileSync(path.join(__dirname, '../src/web/index.html'), 'utf8');
const source = html.match(/<script>\s*([\s\S]*?)<\/script>/)[1];
new vm.Script(source);

const elements = new Map();
const images = [];
const drawing = new Proxy({}, {
  get: (_, name) => name === 'measureText' ? () => ({ width: 20 }) : () => {},
  set: () => true,
});
function element(id) {
  const classes = new Set();
  return {
    id, value: '0', style: {}, events: {}, width: 1920, height: 1080,
    classList: {
      add: value => classes.add(value),
      remove: value => classes.delete(value),
      toggle(value, enabled) { enabled ? classes.add(value) : classes.delete(value); },
    },
    addEventListener(name, handler) { this.events[name] = handler; },
    appendChild(child) { (this.children ||= []).push(child); },
    getContext() { return drawing; },
    getBoundingClientRect() { return { left: 20, top: 30, width: 960, height: 540 }; },
  };
}
class TestImage {
  constructor() { this.naturalWidth = 1920; this.naturalHeight = 1080; images.push(this); }
  set src(value) { this.url = value; }
}
const response = (data, ok = true) => ({ ok, status: ok ? 200 : 400, json: async () => data });
let fetcher = async () => response({ videos: [] });
const sandbox = {
  document: {
    getElementById(id) { if (!elements.has(id)) elements.set(id, element(id)); return elements.get(id); },
    createElement: element,
  },
  Image: TestImage,
  window: { addEventListener() {} },
  fetch: (...args) => fetcher(...args),
  setInterval: () => 1, clearInterval() {}, setTimeout,
  console, Date, Math, Number, JSON, encodeURIComponent,
};
vm.createContext(sandbox);
// Each test controls requests; suppress only the final page-load invocation.
vm.runInContext(source.replace(/\s*loadVideoList\(\);\s*$/, ''), sandbox);
const run = script => vm.runInContext(script, sandbox);
const corners = [[100, 100], [500, 100], [700, 600], [20, 600]];
const confirmed = { corners, source: 'saved', confirmed: true, calibrated: true, confidence: null };
const proposal = { corners, source: 'saved', confirmed: false, calibrated: true, confidence: 0.8 };
const deferred = () => {
  let resolve;
  const promise = new Promise(done => { resolve = done; });
  return { promise, resolve };
};

(async () => {
  run('currentVideo = "one.mp4"; imageReady = true;');
  sandbox.payload = proposal;
  run('applyCalibration(payload); startLiveAnalysis();');
  assert.equal(run('courtConfirmed'), false, 'saved proposals require explicit confirmation');
  assert.equal(run('isAnalyzing'), false, 'analysis cannot start with a proposal');
  assert.equal(elements.get('startLiveBtn').disabled, true);
  run('applyCalibration(payload);');
  assert.match(elements.get('courtStatusText').textContent, /line match 80%/);

  fetcher = async () => response({ detail: 'Invalid court' }, false);
  run('isCalibrationMode = true;');
  await run('saveCorners();');
  assert.equal(run('courtConfirmed'), false);
  assert.equal(run('isCalibrationMode'), true, 'failed saves keep the editor open');
  assert.match(elements.get('courtStatusText').textContent, /not saved/);

  const legacy = { status: 'success', file: 'court.json', corners };
  sandbox.payload = legacy;
  assert.throws(() => run('applyCalibration(payload);'), /server is running an older version/i,
    'legacy reads must explain the backend mismatch before allowing an old save');
  fetcher = async () => response(legacy);
  assert.equal(await run('confirmAndLockCourt();'), false);
  assert.equal(run('courtConfirmed'), false, 'an old successful response cannot unlock analysis');
  assert.match(elements.get('confirmHelp').textContent, /restart the app with python app.py/i);

  fetcher = async () => response({ ...proposal, status: 'success' });
  assert.equal(await run('confirmAndLockCourt();'), false);
  assert.equal(run('courtConfirmed'), false, 'explicit false confirmation must remain rejected');
  fetcher = async () => response(confirmed);
  assert.equal(await run('confirmAndLockCourt();'), true);
  assert.equal(elements.get('startLiveBtn').disabled, false);

  elements.get('calibCanvas').events.mousedown({ clientX: 70, clientY: 80 });
  elements.get('calibCanvas').events.mousemove({ clientX: 80, clientY: 90 });
  assert.equal(run('clickedCorners[0].x'), 120);
  assert.equal(run('courtConfirmed'), false, 'dragging invalidates confirmation');
  assert.equal(elements.get('startLiveBtn').disabled, true);

  fetcher = async () => response({ detail: 'No court found' }, false);
  await run('triggerAutoDetectCourt();');
  assert.match(elements.get('courtStatusText').textContent, /failed/);
  assert.equal(run('courtConfirmed'), false);

  sandbox.payload = confirmed;
  run('applyCalibration(payload); startLiveAnalysis();');
  assert.equal(run('isAnalyzing'), true);
  fetcher = async url => response(url === '/api/live_stats'
    ? { error: 'Unable to initialize analysis', video_name: 'one.mp4', current_frame: 0, is_running: false }
    : {});
  await elements.get('liveFeedImg').onerror();
  assert.equal(run('isAnalyzing'), false, 'stream failure restores local controls');
  assert.equal(elements.get('liveStatusBadge').textContent, 'ANALYSIS FAILED');

  const first = deferred(), second = deferred();
  fetcher = url => url.includes('one.mp4') ? first.promise : second.promise;
  const firstLoad = run('onVideoChange("one.mp4");');
  const secondLoad = run('onVideoChange("two.mp4");');
  second.resolve(response({ ...proposal, source: 'auto_detected' }));
  await secondLoad;
  first.resolve(response({ ...confirmed, corners: [[1, 1], [2, 1], [2, 2], [1, 2]] }));
  await firstLoad;
  assert.equal(run('currentVideo'), 'two.mp4');
  assert.equal(run('clickedCorners[0].x'), 100, 'stale responses cannot overwrite the selected court');
  assert.equal(run('courtConfirmed'), false);

  const latest = images.at(-1), stale = images.at(-2);
  latest.naturalWidth = 1000; latest.naturalHeight = 1000; latest.onload(); stale.onload();
  assert.equal(elements.get('monitorViewport').style.aspectRatio, '1000 / 1000');
  assert.equal(run('canvasPoint({ clientX: 500, clientY: 300 }).y'), 500);
  elements.get('startFrame').value = '300';
  run('loadInitialFrame();');
  assert.match(images.at(-1).url, /frame_idx=0$/, 'calibration preview always uses its opening-frame anchor');

  run('imageReady = true;');
  sandbox.payload = confirmed;
  run('applyCalibration(payload); startLiveAnalysis();');
  fetcher = async () => response({ video_name: 'two.mp4', court_tracking_ok: false,
    court_tracking_error: 'Camera alignment could not be verified', is_running: true,
    current_frame: 1, current_speed_mph: 0, peak_speed_mph: 0, current_shot: 'Rally', players: {} });
  await run('updateLiveStats();');
  assert.equal(elements.get('courtStatusText').textContent, 'Court tracking uncertain');
  fetcher = async () => response({ video_name: 'two.mp4', court_tracking_ok: true,
    is_running: true, current_frame: 2, current_speed_mph: 0, peak_speed_mph: 0,
    current_shot: 'Rally', players: {} });
  await run('updateLiveStats();');
  assert.equal(elements.get('courtStatusText').textContent, 'Court tracking restored');
  assert.equal(elements.get('liveSpeedReadout').textContent, '--', 'missing 3D speed is not zero MPH');

  let finishPolls = 0, resultRequests = 0;
  fetcher = async url => {
    if (url === '/api/stop') return response({ status: 'stopping' });
    if (url === '/api/live_stats') {
      finishPolls++;
      assert.equal(run('isAnalyzing'), true, 'keep controls locked during reconstruction');
      return response({ is_running: finishPolls === 1, video_name: 'two.mp4' });
    }
    resultRequests++;
    assert.equal(finishPolls, 2, 'do not read stale results before reconstruction finishes');
    return response({ summary: { speed_method: 'monocular_3d_flight_v1', max_rally_speed_mph: 25.3,
                                total_flight_segments: 1, speed_note: 'Experimental estimate' },
                      shots: [{ shot_id: 1, shot_type: 'Flight segment', avg_mph: 20.2,
                                speed_status: 'estimated', speed_confidence: .7 }] });
  };
  await run('stopLiveAnalysis();');
  assert.equal(resultRequests, 1);
  assert.equal(run('isAnalyzing'), false);
  assert.equal(elements.get('liveSpeedReadout').textContent, '20.2');
  assert.equal(elements.get('livePeakReadout').textContent, '25.3 MPH');
  fetcher = async () => response({
    summary: { speed_method: 'monocular_3d_flight_v1', max_rally_speed_mph: null,
               total_flight_segments: 33,
               speed_display_reason: 'MPH unavailable: none of 33 flight fragments passed the checks.' },
    shots: [{ shot_id: 1, shot_type: 'Unclassified', avg_mph: null,
              speed_status: 'unavailable', speed_reason: 'Insufficient observations' }],
  });
  await run('loadFinalResults();');
  assert.equal(elements.get('liveSpeedReadout').textContent, '--');
  assert.equal(elements.get('livePeakReadout').textContent, '--');
  assert.match(elements.get('speedStatus').textContent, /none of 33/);
  assert.match(elements.get('shotTableBody').children.at(-1).innerHTML, /Insufficient observations/,
    'rejection reasons must be visible in the row, not only on hover');
  console.log('PASS: calibration save/confirmation, dirty gating, failures, request races, pixel mapping, anchor frame, and motion warnings.');
})().catch(error => { console.error(error); process.exitCode = 1; });
