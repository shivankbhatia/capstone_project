const els = {
  keyboard: document.querySelector('#keyboard'),
  start: document.querySelector('#startButton'),
  reset: document.querySelector('#resetButton'),
  speed: document.querySelector('#speedSelect'),
  source: document.querySelector('#sourceSelect'),
  arm: document.querySelector('#armSelect'),
  scanModeDescription: document.querySelector('#scanModeDescription'),
  samples: document.querySelector('#sampleSelect'),
  custom: document.querySelector('#customMessage'),
  targetLabel: document.querySelector('#targetLabel'),
  targetText: document.querySelector('#targetText'),
  intentLength: document.querySelector('#intentLength'),
  intentSource: document.querySelector('#intentSource'),
  evidenceNote: document.querySelector('#evidenceNote'),
  connection: document.querySelector('#connectionState'),
  decoded: document.querySelector('#decodedText'),
  progress: document.querySelector('#progressText'),
  bar: document.querySelector('#progressBar'),
  phase: document.querySelector('#phaseBadge'),
  stimulus: document.querySelector('#currentStimulus'),
  evidence: document.querySelector('#evidenceScore'),
  flashes: document.querySelector('#flashCount'),
  scan: document.querySelector('#scanCopy'),
  signal: document.querySelector('#signalValue'),
  signalLabel: document.querySelector('#signalLabel'),
  marker: document.querySelector('#p300Marker'),
  state: document.querySelector('#decodeState'),
  confidence: document.querySelector('#confidenceBadge'),
  lastChar: document.querySelector('#lastChar'),
  lastLabel: document.querySelector('#lastLabel'),
  lastDetail: document.querySelector('#lastDetail'),
  events: document.querySelector('#eventList'),
  context: document.querySelector('#contextText'),
  explanation: document.querySelector('#contextExplanation'),
  chips: document.querySelector('#predictionChips'),
  prior: document.querySelector('#priorConfidence'),
  priority: document.querySelector('#priorityTarget'),
  priorityDesc: document.querySelector('#priorityDescription'),
  provenance: document.querySelector('#replayProvenance'),
  ragOptions: document.querySelector('#speedDialOptions'),
  ragStatus: document.querySelector('#speedDialStatus'),
  bars: ['#barOne', '#barTwo', '#barThree', '#barFour'].map(s => document.querySelector(s)),
  canvas: document.querySelector('#waveCanvas'),
  focusToggle: document.querySelector('#focusToggle'),
  focusOverlay: document.querySelector('#flashFocusOverlay'),
  focusLabel: document.querySelector('#flashFocusLabel'),
  focusKeys: document.querySelector('#flashFocusKeys'),
};
const state = {
  result: null,
  selection: 0,
  trace: 0,
  flash: 0,
  timer: null,
  running: false,
  events: [],
  evidence: 0,
  focusEnabled: false,
  request: null,
  requestId: 0
};
const traceCache = new Map();
const displayKey = key => key === 'Sp' ? 'SPACE' : key === 'Prd' ? '.' : key;
const displayChar = char => char === ' ' ? '␠' : char;
const escapeHtml = value => value.replace(/[&<>]/g, char => ({
  '&': '&amp;', '<': '&lt;', '>': '&gt;'
})[char]);
const cell = key => document.querySelector(`.key[data-key="${CSS.escape(key)}"]`);
function setPhase(text, active = false) {
  els.phase.innerHTML = `<span></span> ${text}`;
  els.phase.classList.toggle('active', active);
}
function addEvent(message, emphasis = '') {
  const now = new Date().toLocaleTimeString([], {
    minute: '2-digit', second: '2-digit'
  });
  state.events.unshift({
    now, message, emphasis
  });
  state.events = state.events.slice(0, 5);
  els.events.innerHTML = state.events.map(e => `<div class="event"><time>${e.now}</time>${e.emphasis ? `<strong>${
    e.emphasis
  }
  </strong> ` : ''}${e.message}</div>`).join('');
}
function clearGrid() {
  document.querySelectorAll('.key').forEach(node => node.classList.remove('priority', 'target', 'flashing', 'eeg-detected'));
}
function clearFlashHighlights() {
  document.querySelectorAll('.key').forEach(node => node.classList.remove('flashing', 'eeg-detected'));
}
function closeFlashFocus() {
  if (els.focusOverlay.open) els.focusOverlay.close();
}
function showFlashFocus(flash) {
  if (!state.focusEnabled) return closeFlashFocus();
  els.focusLabel.textContent = `${flash.label} · CURRENT FLASH`;
  els.focusKeys.innerHTML = '';
  flash.keys.forEach(key => {
    const node = document.createElement('div'); node.className = 'flash-focus-key'; node.textContent = displayKey(key); els.focusKeys.appendChild(node);
  });
  if (!els.focusOverlay.open) els.focusOverlay.showModal();
}
function toggleFlashFocus(force) {
  state.focusEnabled = typeof force === 'boolean' ? force : !state.focusEnabled;
  els.focusToggle.setAttribute('aria-pressed', String(state.focusEnabled));
  els.focusToggle.textContent = state.focusEnabled ? 'Exit flash focus' : 'Focus current flash';
  if (!state.focusEnabled) closeFlashFocus();
}
function updateArm() {
  const fixedLm = els.arm.value === 'fixed_lm';
  els.scanModeDescription.textContent = fixedLm ? 'Fixed LM fusion α=.85; clean-M0 ablation ladder lock; τ=.80, min 2, max 10.' : 'EEG-only arm; τ=.80, min 2, max 10.';
}
function appendKeyboardKey(key) {
  if (els.source.value === 'replay' || state.running) return;
  const value = key === 'Sp' ? ' ' : key === 'Prd' ? '.' : key.length === 1 ? key : '';
  if (!value) return;
  els.custom.value = `${els.custom.value}${value}`.slice(0, 80);
  reset();
  updateTargetPreview();
}
function buildKeyboard(keys) {
  els.keyboard.innerHTML = '';
  keys.forEach(key => {
    const button = document.createElement('button'); button.type = 'button'; button.className = `key${key.length > 1 ? ' special' : ''}`; button.dataset.key = key; button.textContent = displayKey(key); button.setAttribute('aria-label', `Add ${displayKey(key)} to simulated text`); button.addEventListener('click', () => appendKeyboardKey(key)); els.keyboard.appendChild(button);
  });
}
function currentTarget() {
  return els.source.value === 'replay' ? els.samples.selectedOptions[0]?.dataset.target || '' : els.source.value === 'aac' ? 'I NEED WATER' : els.custom.value.toUpperCase();
}
function updateTargetPreview() {
  const target = currentTarget(),
  source = els.source.value === 'replay' ? 'held-out Study D replay' : els.source.value === 'aac' ? 'AAC simulation preset' : 'custom simulated EEG';
  els.targetText.textContent = target || 'Choose a sample';
  els.intentLength.textContent = `${target.length} characters`;
  els.intentSource.textContent = source;
}
function setSourceMode() {
  const replay = els.source.value === 'replay',
  aac = els.source.value === 'aac';
  els.samples.hidden = !replay;
  els.custom.hidden = replay || aac;
  els.targetLabel.textContent = replay ? 'STUDY D SAMPLE' : aac ? 'AAC PRESET' : 'CUSTOM SENTENCE';
  reset();
  updateTargetPreview();
  els.evidenceNote.textContent = replay ? 'Choose a held-out session to replay its recorded EEG classifier evidence.' : aac ? 'AAC communication preset uses simulated target-conditioned EEG likelihoods; no recorded EEG or participant result is claimed.' : 'Custom text uses simulated target-conditioned EEG likelihoods; no recorded EEG is claimed.';
  els.signalLabel.textContent = replay ? 'EEG / P300 RESPONSE' : 'SIMULATED EEG / DECODER POSTERIOR';
}
function renderPrior(selection) {
  const top = selection.prior.top || [];
  const fixedLm = selection.prior.arm === 'fixed_lm';
  document.querySelectorAll('.key').forEach(node => node.classList.remove('priority'));
  top.forEach(item => cell(item.key)?.classList.add('priority'));
  els.chips.innerHTML = fixedLm ? top.map((item, i) => `<span class="chip ${i === 0 ? 'top' : ''}">${displayKey(item.key)} <b>${Math.round(item.probability * 100)}%</b></span>`).join('') : '<span class="chip">EEG-only arm: no language-model prior</span>';
  els.prior.textContent = fixedLm ? `top LM prior ${Math.round((top[0]?.probability || 0) * 100)}%` : 'EEG-only';
  els.priority.textContent = 'Uniform replay';
  els.priorityDesc.textContent = 'Recorded order when available; otherwise fixed row then column. RAG is disabled.';
  els.ragOptions.innerHTML = fixedLm ? '<span class="chip">Fixed LM α=.85 · RAG disabled</span>' : '<span class="chip">No language prior</span>';
  els.ragStatus.textContent = fixedLm ? 'Fixed LM prior · α=.85' : 'EEG-only arm';
  els.bars.forEach((bar, i) => {
    bar.style.height = `${Math.max(5, Math.round((top[i]?.probability || 0) * 100))}px`;
  });
}
function render() {
  const result = state.result,
  decoded = result ? result.selections.slice(0, state.selection).map(item => item.predicted_character).join('') : '',
  target = result?.target || currentTarget();
  els.decoded.innerHTML = `${escapeHtml(decoded)}<span class="cursor"></span>`;
  els.progress.textContent = `${state.selection} / ${result?.selections.length || target.length}`;
  els.bar.style.width = `${result?.selections.length ? (state.selection / result.selections.length) * 100 : 0}%`;
  els.context.textContent = decoded || 'No decoded context yet';
  els.explanation.textContent = result?.arm === 'fixed_lm' ? 'The fixed LM prior is computed from decoder output, not the intended target.' : 'EEG-only arm: no language-model prior is applied.';
  els.provenance.textContent = result ? `${result.summary.mean_sequences_per_character} sequences / char` : '—';
}
function drawWave() {
  const canvas = els.canvas,
  ctx = canvas.getContext('2d'),
  dpr = devicePixelRatio || 1,
  width = canvas.clientWidth,
  height = canvas.clientHeight;
  canvas.width = width * dpr;
  canvas.height = height * dpr;
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, width, height);
  ctx.strokeStyle = '#e5ece5';
  ctx.lineWidth = 1;
  for (let y = 20; y < height; y += 27) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
    ctx.stroke();
  }
  ctx.beginPath();
  for (let x = 0; x < width; x += 1) {
    const baseline = Math.sin(x * .085) * 3 + Math.sin(x * .028) * 2,
    peak = Math.exp(-Math.pow((x - width * .34) / 34, 2)) * state.evidence * 35,
    y = height * .55 + baseline - peak;
    x ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
  }
  ctx.strokeStyle = '#278359';
  ctx.lineWidth = 1.7;
  ctx.stroke();
}
async function fetchFlashTrace(sessionId, charIndex, sequence, stimulusCode) {
  const key = `${sessionId}:${charIndex}:${sequence}:${stimulusCode}`;
  if (traceCache.has(key)) return traceCache.get(key);
  const response = await fetch(`/api/flash_trace?session_id=${encodeURIComponent(sessionId)}&char_index=${charIndex}&sequence=${sequence}&stimulus_code=${stimulusCode}`);
  if (!response.ok) return null;
  const trace = await response.json();
  traceCache.set(key, trace);
  return trace;
}
function drawRealWave(trace) {
  const canvas = els.canvas,
  ctx = canvas.getContext('2d'),
  dpr = devicePixelRatio || 1,
  width = canvas.clientWidth,
  height = canvas.clientHeight;
  canvas.width = width * dpr;
  canvas.height = height * dpr;
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, width, height);
  ctx.strokeStyle = '#e5ece5';
  ctx.lineWidth = 1;
  for (let y = 20; y < height; y += 27) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
    ctx.stroke();
  }
  const amplitudes = trace.amplitude_uv,
  count = amplitudes.length,
  maxAbs = Math.max(1, ...amplitudes.map(Math.abs));
  ctx.beginPath();
  amplitudes.forEach((value, index) => {
    const x = (index / Math.max(1, count - 1)) * width, y = height * .55 - (value / maxAbs) * (height * .4); index ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
  });
  ctx.strokeStyle = '#278359';
  ctx.lineWidth = 1.7;
  ctx.stroke();
}
function showFlash() {
  if (!state.running || !state.result) return;
  const selection = state.result.selections[state.selection],
  trace = selection.trace[state.trace];
  if (!trace) return confirmSelection();
  const flash = trace.flashes[state.flash];
  if (!flash) {
    state.trace += 1;
    state.flash = 0;
    return showFlash();
  }
  clearFlashHighlights();
  flash.keys.forEach(key => cell(key)?.classList.add('flashing'));
  showFlashFocus(flash);
  state.evidence = flash.target_probability;
  if (els.source.value === 'replay' && state.result.session_id) {
    fetchFlashTrace(state.result.session_id, state.selection, trace.sequence, flash.stimulus_code).then(traceData => traceData ? drawRealWave(traceData) : drawWave());
  }
  else {
    drawWave();
  }
  els.marker.classList.toggle('show', flash.classifier_target);
  els.signal.textContent = flash.target_probability.toFixed(2);
  els.evidence.textContent = flash.target_probability.toFixed(2);
  els.stimulus.textContent = `${flash.label} · ${trace.flash_mode === 'simulated_order' ? 'SIMULATED RC ORDER' : 'RECORDED RC ORDER'}`;
  els.flashes.textContent = `${trace.sequence} / 10`;
  els.scan.textContent = `${flash.label} flashed · classifier target ${Math.round(flash.target_probability * 100)}%`;
  els.state.textContent = 'Flashing row/column and scoring EEG';
  if (flash.classifier_target) {
    flash.keys.forEach(key => cell(key)?.classList.add('eeg-detected'));
    addEvent(`${flash.label} registered as a classifier target (${Math.round(flash.target_probability * 100)}%).`, 'P300 detected');
  }
  state.flash += 1;
  state.timer = setTimeout(showFlash, 175 / Number(els.speed.value));
}
function beginSelection() {
  if (!state.running || !state.result) return;
  if (state.selection >= state.result.selections.length) return finish();
  const selection = state.result.selections[state.selection];
  state.trace = 0;
  state.flash = 0;
  clearGrid();
  cell(selection.target_key)?.classList.add('target');
  renderPrior(selection);
  render();
  els.state.textContent = 'Preparing fixed row/column replay';
  els.confidence.textContent = '—';
  addEvent(`Character ${selection.index + 1}: recorded order when available; otherwise fixed row then column.`, 'Replay ready');
  showFlash();
}
function confirmSelection() {
  closeFlashFocus();
  const selection = state.result.selections[state.selection];
  clearTimeout(state.timer);
  cell(selection.predicted_key)?.classList.add('eeg-detected');
  const predicted = displayChar(selection.predicted_character);
  els.lastChar.textContent = predicted;
  els.lastLabel.textContent = `${predicted} ${selection.correct ? 'matches recorded target' : 'does not match recorded target'}`;
  els.lastDetail.textContent = `${selection.sequences} sequence(s) · decoder confidence ${Math.round(selection.confidence * 100)}%`;
  els.confidence.textContent = `${Math.round(selection.confidence * 100)}% confidence`;
  els.state.textContent = 'Decoder selection confirmed';
  addEvent(`Decoder selected ${displayKey(selection.predicted_key)} after ${selection.sequences} sequence(s).`, selection.correct ? 'Confirmed' : 'Mismatch');
  state.selection += 1;
  render();
  state.timer = setTimeout(beginSelection, 620 / Number(els.speed.value));
}
function finish() {
  state.running = false;
  clearTimeout(state.timer);
  setPhase('Backend run complete');
  els.start.innerHTML = '✓ Run complete';
  els.start.classList.add('running');
  els.scan.textContent = `Single-run accuracy ${state.result.summary.accuracy}% · not a population estimate`;
  els.state.textContent = 'Backend result reproduced';
  addEvent('Frozen replay result displayed; not live acquisition or a population estimate.', 'Complete');
}
function reset({ cancelRequest = true, invalidateRequest = true } = {}) {
  if (cancelRequest) state.request?.abort();
  state.request = null;
  if (invalidateRequest) state.requestId += 1;
  toggleFlashFocus(false);
  state.running = false;
  state.result = null;
  state.selection = 0;
  state.trace = 0;
  state.flash = 0;
  state.evidence = 0;
  state.events = [];
  clearTimeout(state.timer);
  clearGrid();
  els.events.innerHTML = '';
  els.chips.innerHTML = '';
  els.ragOptions.innerHTML = '';
  els.ragStatus.textContent = 'Fixed LM status appears here';
  els.stimulus.textContent = '—';
  els.evidence.textContent = '0.00';
  els.flashes.textContent = '0';
  els.signal.textContent = '0.0';
  els.marker.classList.remove('show');
  els.scan.textContent = 'Load a backend run to inspect its row/column flash trace';
  els.lastChar.textContent = '—';
  els.lastLabel.textContent = 'No selection yet';
  els.lastDetail.textContent = 'Confidence will appear here.';
  els.confidence.textContent = '—';
  els.state.textContent = 'Waiting for backend evidence';
  els.priority.textContent = 'Uniform replay';
  els.priorityDesc.textContent = 'Recorded order when available; otherwise fixed row then column.';
  els.prior.textContent = '—';
  els.bars.forEach(bar => {
    bar.style.height = '0px';
  });
  els.start.innerHTML = '<span class="play-icon">▶</span> Load backend run';
  els.start.classList.remove('running');
  setPhase('Awaiting backend run');
  render();
  drawWave();
}
async function loadRun() {
  if (state.running) {
    state.running = false;
    clearTimeout(state.timer);
    els.start.innerHTML = '<span class="play-icon">▶</span> Resume run';
    els.start.classList.remove('running');
    setPhase('Playback paused');
    return;
  }
  if (state.result && state.selection < state.result.selections.length) {
    state.running = true;
    els.start.innerHTML = '<span class="play-icon">Ⅱ</span> Pause playback';
    els.start.classList.add('running');
    setPhase(state.result.source === 'study_d_replay' ? 'Replay in progress' : 'Simulation in progress', true);
    return showFlash();
  }
  const replay = els.source.value === 'replay',
  aac = els.source.value === 'aac',
  payload = replay ? {
    session_id: els.samples.value,
    arm: els.arm.value
  }
  : {
    message: aac ? 'I NEED WATER' : els.custom.value,
    arm: els.arm.value,
    scenario: aac ? 'aac_simulation' : 'custom'
  },
  requestId = state.requestId + 1,
  controller = new AbortController();
  state.requestId = requestId;
  state.request = controller;
  els.start.disabled = true;
  els.start.textContent = els.arm.value === 'fixed_lm' ? 'Loading local language model…' : 'Running backend…';
  setPhase(els.arm.value === 'fixed_lm' ? 'Loading fixed LM' : 'Backend is decoding', true);
  try {
    const response = await fetch(replay ? '/api/replay' : '/api/simulate', {
      method: 'POST', headers: {
        'Content-Type': 'application/json'
      }, body: JSON.stringify(payload), signal: controller.signal
    }),
    result = await response.json();
    if (requestId !== state.requestId) return;
    if (!response.ok) throw new Error(result.error || 'Backend request failed');
    state.request = null;
    reset({
      cancelRequest: false,
      invalidateRequest: false
    });
    state.result = result;
    els.targetText.textContent = result.target;
    els.intentLength.textContent = `${result.summary.characters} characters`;
    const recorded = result.source.startsWith('study_d');
    els.intentSource.textContent = recorded ? 'recorded Study D data' : aac ? 'AAC simulation preset' : 'custom simulation';
    els.evidenceNote.textContent = result.evidence_notice;
    els.signalLabel.textContent = recorded ? 'RECORDED EEG / SWLDA POSTERIOR' : 'SIMULATED EEG / DECODER POSTERIOR';
    state.running = true;
    els.start.innerHTML = '<span class="play-icon">Ⅱ</span> Pause playback';
    els.start.classList.add('running');
    setPhase(recorded ? 'Replay in progress' : 'Simulation in progress', true);
    addEvent(result.evidence_notice, 'Backend ready');
    beginSelection();
  }
  catch (error) {
    if (error.name !== 'AbortError' && requestId === state.requestId) {
      setPhase('Backend error');
      els.scan.textContent = error.message;
      addEvent(error.message, 'Error');
    }
  }
  finally {
    if (requestId === state.requestId) {
      state.request = null;
      els.start.disabled = false;
      if (!state.running) els.start.innerHTML = '<span class="play-icon">▶</span> Load backend run';
    }
  }
}
async function initialise() {
  try {
    const [healthResponse, samplesResponse] = await Promise.all([fetch('/api/health'), fetch('/api/samples')]),
    health = await healthResponse.json(),
    samples = await samplesResponse.json();
    if (!healthResponse.ok || !samplesResponse.ok) throw new Error(health.error || samples.error || 'Backend unavailable');
    buildKeyboard(health.grid.keys);
    const fixedLmOption = els.arm.querySelector('option[value="fixed_lm"]');
    fixedLmOption.disabled = !health.fixed_lm_available_offline;
    if (!health.fixed_lm_available_offline && els.arm.value === 'fixed_lm') els.arm.value = 'eeg_only';
    els.samples.innerHTML = samples.samples.length ? samples.samples.map(sample => `<option value="${sample.id}" data-target="${escapeHtml(sample.target)}">${sample.id} · ${sample.target}</option>`).join('') : '<option value="">No Study D replay samples available</option>';
    els.connection.innerHTML = '<i></i> Backend connected';
    els.connection.title = health.custom_text_notice;
    updateTargetPreview();
    reset();
    updateArm();
    if (!samples.samples.length) {
      els.scan.textContent = health.using_fallback_grid ? 'No processed Study D data found. Custom simulation is available on a built-in demo keyboard.' : health.heldout_evaluation_enabled ? 'No replay sessions found. Check data/processed/StudyD and the held-out session manifest.' : 'Held-out targets are hidden. Restart with --allow-heldout-eval to load Study D replay samples.';
      els.evidenceNote.textContent = health.fixed_lm_available_offline ? 'Study D replay requires the explicit held-out evaluation option and processed data. Custom text is labelled simulated.' : 'Fixed LM is unavailable offline; EEG-only custom simulation remains available and is labelled simulated.';
    }
  }
  catch (error) {
    els.connection.textContent = 'Backend unavailable';
    els.connection.classList.add('offline');
    els.start.disabled = true;
    els.scan.textContent = 'Start prototype/server.py to connect the real backend.';
  }
}
els.start.addEventListener('click', loadRun);
els.reset.addEventListener('click', reset);
els.focusToggle.addEventListener('click', () => toggleFlashFocus());
els.focusOverlay.addEventListener('cancel', () => toggleFlashFocus(false));
document.addEventListener('keydown', event => {
  if (event.key === 'Escape') toggleFlashFocus(false);
});
els.source.addEventListener('change', setSourceMode);
els.arm.addEventListener('change', () => {
  reset(); updateArm();
});
els.samples.addEventListener('change', () => {
  reset(); updateTargetPreview();
});
els.custom.addEventListener('input', () => {
  reset(); updateTargetPreview();
});
window.addEventListener('resize', drawWave);
initialise();
