/* SkoliData wear algorithm v1 (calibration + wear detection)
 *
 * The idea in plain words:
 *   The sensor sits against the skin. Brace ON -> it warms toward skin temperature; brace OFF -> it cools
 *   toward room temperature. It never jumps: it follows a curve  T(t) = T_end + (T_start - T_end) * e^(-t/tau),
 *   where tau ("tau", seconds) says how slow the button is.
 *
 *   Calibration = 5 min brace OFF, then 5 min brace ON. For each phase we fit that curve, which lets us
 *   predict where the temperature WOULD settle (T_end) even if 5 minutes was not long enough to get there.
 *     Toff = settled OFF temperature (room), Ton = settled ON temperature (skin under the brace),
 *     gap = Ton - Toff.
 *   Rules: ON when T rises above Toff + 60% of the gap, OFF when T falls below Toff + 40% of the gap
 *   (the 20% band between them stops flickering). Because the curve is slow, the real moment of putting
 *   the brace on was earlier than the crossing: tau_on * ln(1/0.4) ~ 0.92 tau_on earlier. Same for removal.
 *   Room temperature changes during the day, so the OFF level is slowly re-learned while the brace is off,
 *   and a fast fall from the recent ON level also counts as removal (helps in hot rooms).
 *   When room and skin are too close to tell apart, time is marked "not sure" and not counted.
 *
 * Works in the browser (window.SkoliAlgo) and in Node (require).
 */
(function (root) {
  'use strict';

  const CFG = {
    phaseMinSamples: 0.75,     // need >= 75% of the expected readings in each phase
    maxGapS: 20,               // a calibration phase may not have holes longer than this
    minGapC: 2.0,              // ON and OFF must be at least 2 C apart
    tonRange: [28, 38.5],      // plausible skin-under-brace temperature
    toffRange: [0, 34],        // plausible room temperature
    maxNoiseC: 0.35,           // residual noise around the fitted curve
    tauRange: [5, 1200],       // seconds
    onFrac: 0.60, offFrac: 0.40,
    smoothN: 5,                // median of 5 readings (10 s) against spikes
    dataGapS: 60,              // a hole longer than this = "no data" (never counted as worn)
    minSegS: 120,              // ignore ON blips and OFF dips shorter than 2 min
    baselineTauS: 1200,        // OFF level re-learned with a 20 min memory
    unsureGapC: 2.0,           // if (Ton - current OFF level) < 2 C we can't tell -> "not sure"
    dropFrac: 0.25, dropMinC: 1.2, dropMaxC: 2.5, onRefWindowS: 600,
    hotterThanSkinC: 0.6,      // readings this much above the calibrated skin level can't be judged -> not sure
    maxLagS: 600,
  };

  // ---------- small helpers ----------
  const mean = a => a.reduce((s, x) => s + x, 0) / a.length;
  const median = a => { const b = [...a].sort((x, y) => x - y), m = b.length >> 1; return b.length % 2 ? b[m] : (b[m - 1] + b[m]) / 2; };
  const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));

  // Fit T(t) = A + B*exp(-t/tau) by trying many tau values; for each, A and B come from ordinary least squares.
  function fitCurve(ts, ys) {
    const t0 = ts[0], x = ts.map(t => t - t0);
    let best = null;
    for (let k = 0; k <= 80; k++) {
      const tau = CFG.tauRange[0] * Math.pow(CFG.tauRange[1] / CFG.tauRange[0], k / 80);
      const e = x.map(v => Math.exp(-v / tau));
      const n = ys.length, se = e.reduce((s, v) => s + v, 0), sy = ys.reduce((s, v) => s + v, 0);
      const see = e.reduce((s, v) => s + v * v, 0), sey = e.reduce((s, v, i) => s + v * ys[i], 0);
      const det = n * see - se * se;
      if (Math.abs(det) < 1e-9) continue;
      const B = (n * sey - se * sy) / det, A = (sy - B * se) / n;
      let sse = 0;
      for (let i = 0; i < n; i++) { const r = ys[i] - (A + B * e[i]); sse += r * r; }
      if (!best || sse < best.sse) best = { A, B, tau, sse };
    }
    const flat = { A: mean(ys), B: 0, tau: NaN, sse: ys.reduce((s, y) => s + (y - mean(ys)) ** 2, 0) };
    if (!best) best = flat;
    best.rms = Math.sqrt(best.sse / ys.length);
    // if the curve barely moves (< 0.3 C over the phase) tau can't be known -> treat as already settled
    if (Math.abs(best.B) < 0.3) { best.settled = true; best.tauKnown = false; } else best.tauKnown = true;
    return best;
  }

  function cleanPhase(samples, phaseSeconds, everyS) {
    const ok = samples.filter(s => s.skin !== null && isFinite(s.skin)).sort((a, b) => a.t - b.t);
    const expected = phaseSeconds / everyS;
    let maxGap = 0;
    for (let i = 1; i < ok.length; i++) maxGap = Math.max(maxGap, (ok[i].t - ok[i - 1].t) / 1000);
    return { ok, enough: ok.length >= expected * CFG.phaseMinSamples, maxGap };
  }

  // Settled temperature: the fitted end value, but never further than 2.5 C from the last minute's average
  function settledTemp(fit, ys, ts) {
    const tEnd = ts[ts.length - 1];
    const last = ys.filter((_, i) => tEnd - ts[i] <= 60);
    const lastMean = mean(last.length ? last : ys);
    return { Tinf: fit.settled ? lastMean : clamp(fit.A, lastMean - 2.5, lastMean + 2.5), lastMean };
  }

  /**
   * calibrate(offSamples, onSamples, opts)
   *   samples: [{t: ms, skin: C|null, rh: %|null, chip: C}], from the 5 min OFF and the 5 min ON phases.
   * Returns { ok, reason, model } ; reason is a short code + Turkish text for the screen.
   */
  function calibrate(offSamples, onSamples, opts = {}) {
    const phaseS = opts.phaseSeconds || 300, everyS = opts.everyS || 2;
    const off = cleanPhase(offSamples, phaseS, everyS), on = cleanPhase(onSamples, phaseS, everyS);
    const fail = (code, tr, extra) => ({ ok: false, code, reason: tr, details: extra || {} });
    if (!off.enough || !on.enough) return fail('few_samples', 'Yeterli ölçüm alınamadı. Telefon butona yakın kalmalı.', { off: off.ok.length, on: on.ok.length });
    if (off.maxGap > CFG.maxGapS || on.maxGap > CFG.maxGapS) return fail('gaps', 'Ölçümlerde boşluk var. Lütfen tekrar dene.', { offGap: off.maxGap, onGap: on.maxGap });

    const offT = off.ok.map(s => s.t / 1000), offY = off.ok.map(s => s.skin);
    const onT = on.ok.map(s => s.t / 1000), onY = on.ok.map(s => s.skin);
    const fOff = fitCurve(offT, offY), fOn = fitCurve(onT, onY);
    const sOff = settledTemp(fOff, offY, offT), sOn = settledTemp(fOn, onY, onT);
    const Toff = sOff.Tinf, Ton = sOn.Tinf, gap = Ton - Toff;
    const det = { Toff: +Toff.toFixed(2), Ton: +Ton.toFixed(2), gap: +gap.toFixed(2), tauOn: fOn.tau, tauOff: fOff.tau, rmsOn: fOn.rms, rmsOff: fOff.rms };

    if (fOn.rms > CFG.maxNoiseC || fOff.rms > CFG.maxNoiseC) return fail('noisy', 'Ölçümler çok dalgalı. Kalibrasyon sırasında korseyi oynatma.', det);
    if (Ton < CFG.tonRange[0] || Ton > CFG.tonRange[1]) return fail('ton_range', 'Korse takılıyken beklenen ısınma görülmedi. Korsenin tam oturduğundan emin ol.', det);
    if (Toff < CFG.toffRange[0] || Toff > CFG.toffRange[1]) return fail('toff_range', 'Oda ölçümü beklenmedik. Butonu ten ve güneşten uzak tut.', det);
    if (gap < CFG.minGapC) return fail('small_gap', 'Takılı ve çıkarılmış hali ayırt edilemedi (oda çok sıcak olabilir). Serin bir yerde tekrar dene.', det);
    if (sOn.lastMean < Toff + CFG.onFrac * gap) return fail('not_warmed', 'Korse takılıyken buton yeterince ısınmadı. Lütfen tekrar dene.', det);
    if (sOff.lastMean > Toff + CFG.offFrac * gap + 0.5) return fail('not_cooled', 'Korse çıkarılmışken buton soğumadı. Butonu tenden uzak tut.', det);

    const tauOn = fOn.tauKnown ? clamp(fOn.tau, 10, 900) : 90;
    const tauOff = fOff.tauKnown ? clamp(fOff.tau, 10, 900) : tauOn * 1.5;
    const lastRh = (arr) => { const r = arr.ok.slice(-60).map(s => s.rh).filter(v => v !== null && isFinite(v)); return r.length ? mean(r) : null; };
    const model = {
      version: 1, createdAt: opts.now || Date.now(),
      Toff, Ton, gap,
      thrOn: Toff + CFG.onFrac * gap, thrOff: Toff + CFG.offFrac * gap,
      tauOn, tauOff,
      lagOn: Math.min(CFG.maxLagS, tauOn * Math.log(1 / (1 - CFG.onFrac))),   // time from putting on to crossing thrOn
      lagOff: Math.min(CFG.maxLagS, tauOff * Math.log(1 / CFG.offFrac)),     // time from taking off to crossing thrOff
      rhOff: lastRh(off), rhOn: lastRh(on),
      noise: Math.max(fOn.rms, fOff.rms),
    };
    return { ok: true, code: 'ok', reason: 'Kalibrasyon tamamlandı.', model, details: det };
  }

  /**
   * classify(samples, model)
   *   samples: [{t: ms, skin: C|null}] any length, any order.
   *   Returns per-sample states and totals: state = 'on' | 'off' | 'unsure' | 'nodata' (for the time AFTER a sample).
   */
  function classify(samples, model) {
    const s = samples.filter(x => x.skin !== null && isFinite(x.skin)).sort((a, b) => a.t - b.t);
    const n = s.length, out = new Array(n);
    if (!n) return { states: [], segments: [], wornMs: 0, unsureMs: 0, offMs: 0, nodataMs: 0 };

    // 1) spike filter: running median of 5
    const sm = s.map((_, i) => median(s.slice(Math.max(0, i - (CFG.smoothN >> 1)), Math.min(n, i + (CFG.smoothN >> 1) + 1)).map(x => x.skin)));

    // 2) state machine with a slowly re-learned OFF level
    let base = model.Toff;                    // current OFF (room) level
    let state = sm[0] >= (model.thrOn + model.thrOff) / 2 ? 'on' : 'off';
    let win = [];                             // recent ON readings for a sliding 10-min maximum
    const resetWin = (t, T) => { win = [{ t, T }]; };
    resetWin(s[0].t, sm[0]);
    const gapNow = () => model.Ton - base;
    const flips = [];                         // {i, to}
    for (let i = 0; i < n; i++) {
      const T = sm[i], t = s[i].t;
      const dtS = i ? (t - s[i - 1].t) / 1000 : 0;
      const newBlock = i > 0 && dtS > CFG.dataGapS;
      if (newBlock) { state = T >= base + CFG.onFrac * gapNow() ? 'on' : 'off'; resetWin(t, T); }
      const g = Math.max(gapNow(), 0.01);
      const thrOn = base + CFG.onFrac * g, thrOff = base + CFG.offFrac * g;
      if (state === 'off') {
        if (T >= thrOn) { state = 'on'; resetWin(t, T); flips.push({ i, to: 'on' }); }
        else if (T < base) base = Math.max(T, model.Toff - 8);   // cooling: the room is at most this warm
        else if (dtS > 0) {                   // room warming: learn slowly (a re-wear still crosses thrOn)
          const a = 1 - Math.exp(-dtS / CFG.baselineTauS);
          base = clamp(base + a * (T - base), model.Toff - 8, model.Ton - 1.0);
        }
      } else {
        while (win.length && win[win.length - 1].T <= T) win.pop();     // keep a falling list: front = max
        win.push({ t, T });
        while (win[0].t < t - CFG.onRefWindowS * 1000) win.shift();
        const onRef = win[0].T;
        const drop = clamp(CFG.dropFrac * model.gap, CFG.dropMinC, CFG.dropMaxC);
        if (T <= thrOff || T <= onRef - drop) {
          state = 'off'; flips.push({ i, to: 'off' });
          base = Math.min(base > T ? base : T, model.Ton - 1.0);   // room is at most the current reading
        }
      }
      out[i] = { t, T, state, unsure: gapNow() < CFG.unsureGapC || T > model.Ton + CFG.hotterThanSkinC, gapAfter: i < n - 1 ? (s[i + 1].t - t) / 1000 > CFG.dataGapS : false };
    }

    // 3) back-date each flip by the sensor delay (the real moment was earlier than the crossing)
    //    from the curve: lag = tau * ln((end - start) / (end - T_at_crossing)), using the levels actually seen
    //    in the 10 minutes before and after the crossing (falls back to the calibration delay)
    const W = 600e3;
    const levels = (i, dir) => {             // min/max of smoothed temp before (dir=-1) or after (dir=+1)
      let lo = Infinity, hi = -Infinity;
      for (let j = i; j >= 0 && j < n && Math.abs(s[j].t - s[i].t) <= W; j += dir) {
        if (j !== i && out[dir < 0 ? j : j - 1].gapAfter) break;
        lo = Math.min(lo, sm[j]); hi = Math.max(hi, sm[j]);
      }
      return { lo, hi };
    };
    for (const f of flips) {
      const tc = s[f.i].t, Tc = sm[f.i];
      const before = levels(f.i, -1), after = levels(f.i, +1);
      let lagS;
      if (f.to === 'on') {
        const end = Math.max(after.hi, Tc + 0.2), start = Math.min(before.lo, Tc - 0.2);
        lagS = model.tauOn * Math.log((end - start) / (end - Tc));
      } else {
        const end = Math.min(after.lo, Tc - 0.2), start = Math.max(before.hi, Tc + 0.2);
        lagS = model.tauOff * Math.log((start - end) / (Tc - end));
      }
      if (!isFinite(lagS) || lagS < 0) lagS = f.to === 'on' ? model.lagOn : model.lagOff;
      const lag = Math.min(lagS, CFG.maxLagS) * 1000;
      for (let j = f.i - 1; j >= 0 && s[j].t >= tc - lag && !out[j].gapAfter; j--) out[j].state = f.to;
    }

    // 4) build segments, then remove blips shorter than 2 min (merge into neighbours)
    let segs = [];
    for (let i = 0; i < n; i++) {
      const st = out[i].state, last = segs[segs.length - 1];
      if (last && last.state === st && !out[i - 1].gapAfter) { last.end = i; }
      else segs.push({ state: st, start: i, end: i });
    }
    const segDur = g => (s[g.end].t - s[g.start].t) / 1000 + 2;
    let changed = true;
    while (changed && segs.length > 2) {
      changed = false;
      for (let k = 1; k < segs.length - 1; k++) {
        const a = segs[k - 1], b = segs[k], c = segs[k + 1];
        const joined = !out[a.end].gapAfter && !out[b.end].gapAfter;
        if (joined && a.state === c.state && b.state !== a.state && segDur(b) < CFG.minSegS) {
          for (let i = b.start; i <= b.end; i++) out[i].state = a.state;
          a.end = c.end; segs.splice(k, 2); changed = true; break;
        }
      }
    }

    // 5) totals: each sample stands for the time until the next one (holes > 60 s = no data)
    let wornMs = 0, unsureMs = 0, offMs = 0, nodataMs = 0;
    for (let i = 0; i < n - 1; i++) {
      const dt = s[i + 1].t - s[i].t;
      if (out[i].gapAfter) { nodataMs += dt; continue; }
      if (out[i].unsure) unsureMs += dt;
      else if (out[i].state === 'on') wornMs += dt;
      else offMs += dt;
    }
    segs = segs.map(g => ({ state: g.state, from: s[g.start].t, to: s[g.end].t }));
    return { states: out, segments: segs, wornMs, unsureMs, offMs, nodataMs };
  }

  // Hours per local calendar day (00:00-24:00), from classify() output
  function hoursPerDay(result) {
    const days = {};
    const st = result.states;
    for (let i = 0; i < st.length - 1; i++) {
      if (st[i].gapAfter || st[i].unsure || st[i].state !== 'on') continue;
      let a = st[i].t; const b = st[i + 1].t;
      while (a < b) {
        const d = new Date(a), key = d.getFullYear() + '-' + String(d.getMonth() + 1).padStart(2, '0') + '-' + String(d.getDate()).padStart(2, '0');
        const mid = new Date(d.getFullYear(), d.getMonth(), d.getDate() + 1).getTime();
        const end = Math.min(b, mid);
        days[key] = (days[key] || 0) + (end - a);
        a = end;
      }
    }
    return days;   // ms per day
  }

  const api = { CFG, fitCurve, calibrate, classify, hoursPerDay };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.SkoliAlgo = api;
})(typeof window !== 'undefined' ? window : globalThis);
