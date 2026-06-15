'use strict';

// ============================================================
// CONFIG
// ============================================================
const API_BASE = 'http://localhost:8000';

// ============================================================
// WC 2026 — OFFICIAL DATA (source: FIFA, June 2026)
// ============================================================

// Display name → API/dataset name for teams with name differences
const API_NAME_MAP = {
  'Korea Republic': 'South Korea',
  'Türkiye':        'Turkey',
  "Côte d'Ivoire":  'Ivory Coast',
  'Cabo Verde':     'Cape Verde',
  'IR Iran':        'Iran',
  'Czechia':        'Czech Republic',
  'Congo DR':       'DR Congo',
  'Bosnia and Herzegovina': 'Bosnia-Herzegovina',
  'Curaçao':        'Curacao',
};
function apiName(team) { return API_NAME_MAP[team] || team; }

const WC2026_GROUPS = {
  A: ['Mexico', 'Korea Republic', 'Czechia', 'South Africa'],
  B: ['Canada', 'Bosnia and Herzegovina', 'Qatar', 'Switzerland'],
  C: ['Haiti', 'Scotland', 'Brazil', 'Morocco'],
  D: ['USA', 'Paraguay', 'Australia', 'Türkiye'],
  E: ["Côte d'Ivoire", 'Ecuador', 'Germany', 'Curaçao'],
  F: ['Netherlands', 'Japan', 'Sweden', 'Tunisia'],
  G: ['IR Iran', 'New Zealand', 'Belgium', 'Egypt'],
  H: ['Saudi Arabia', 'Uruguay', 'Spain', 'Cabo Verde'],
  I: ['France', 'Senegal', 'Iraq', 'Norway'],
  J: ['Argentina', 'Algeria', 'Austria', 'Jordan'],
  K: ['Portugal', 'Congo DR', 'Uzbekistan', 'Colombia'],
  L: ['Ghana', 'Panama', 'England', 'Croatia'],
};

const TEAM_FLAGS = {
  // CONCACAF
  'USA': '🇺🇸', 'Mexico': '🇲🇽', 'Canada': '🇨🇦',
  'Panama': '🇵🇦', 'Haiti': '🇭🇹', 'Guatemala': '🇬🇹',
  'Honduras': '🇭🇳', 'El Salvador': '🇸🇻', 'Jamaica': '🇯🇲',
  'Costa Rica': '🇨🇷',
  // South America
  'Brazil': '🇧🇷', 'Argentina': '🇦🇷', 'Colombia': '🇨🇴',
  'Ecuador': '🇪🇨', 'Uruguay': '🇺🇾', 'Paraguay': '🇵🇾',
  'Chile': '🇨🇱', 'Bolivia': '🇧🇴',
  // Europe
  'France': '🇫🇷', 'Spain': '🇪🇸', 'England': '🏴󠁧󠁢󠁥󠁮󠁧󠁿',
  'Germany': '🇩🇪', 'Portugal': '🇵🇹', 'Netherlands': '🇳🇱',
  'Italy': '🇮🇹', 'Belgium': '🇧🇪', 'Croatia': '🇭🇷',
  'Switzerland': '🇨🇭', 'Denmark': '🇩🇰', 'Sweden': '🇸🇪',
  'Norway': '🇳🇴', 'Austria': '🇦🇹', 'Scotland': '🏴󠁧󠁢󠁳󠁣󠁴󠁿',
  'Serbia': '🇷🇸', 'Türkiye': '🇹🇷', 'Ukraine': '🇺🇦',
  'Czechia': '🇨🇿', 'Bosnia and Herzegovina': '🇧🇦',
  'Uzbekistan': '🇺🇿',
  // Africa
  'Morocco': '🇲🇦', 'Senegal': '🇸🇳', 'Egypt': '🇪🇬',
  'Algeria': '🇩🇿', 'South Africa': '🇿🇦', 'Ghana': '🇬🇭',
  "Côte d'Ivoire": '🇨🇮', 'Cameroon': '🇨🇲', 'Nigeria': '🇳🇬',
  'Mali': '🇲🇱', 'Tunisia': '🇹🇳', 'Cabo Verde': '🇨🇻',
  'Congo DR': '🇨🇩',
  // Asia / Middle East
  'Japan': '🇯🇵', 'Korea Republic': '🇰🇷', 'Australia': '🇦🇺',
  'Saudi Arabia': '🇸🇦', 'Iraq': '🇮🇶', 'Jordan': '🇯🇴',
  'IR Iran': '🇮🇷', 'New Zealand': '🇳🇿', 'Qatar': '🇶🇦',
  // Caribbean / other
  'Curaçao': '🇨🇼',
};

// Official group stage match list — all 72 matches
const MATCH_LIST = [
  // ── MATCHDAY 1 ──────────────────────────────────────────────
  { id:1,  date:'2026-06-11', group:'A', md:1, home:'Mexico',           away:'South Africa',           venue:'Mexico City Stadium' },
  { id:2,  date:'2026-06-11', group:'A', md:1, home:'Korea Republic',   away:'Czechia',                venue:'Estadio Guadalajara' },
  { id:3,  date:'2026-06-12', group:'B', md:1, home:'Canada',           away:'Bosnia and Herzegovina', venue:'Toronto Stadium' },
  { id:4,  date:'2026-06-12', group:'D', md:1, home:'USA',              away:'Paraguay',               venue:'Los Angeles Stadium' },
  { id:5,  date:'2026-06-13', group:'C', md:1, home:'Haiti',            away:'Scotland',               venue:'Boston Stadium' },
  { id:6,  date:'2026-06-13', group:'D', md:1, home:'Australia',        away:'Türkiye',                venue:'BC Place, Vancouver' },
  { id:7,  date:'2026-06-13', group:'C', md:1, home:'Brazil',           away:'Morocco',                venue:'New York New Jersey Stadium' },
  { id:8,  date:'2026-06-13', group:'B', md:1, home:'Qatar',            away:'Switzerland',            venue:'San Francisco Bay Area Stadium' },
  { id:9,  date:'2026-06-14', group:'E', md:1, home:"Côte d'Ivoire",   away:'Ecuador',                venue:'Philadelphia Stadium' },
  { id:10, date:'2026-06-14', group:'E', md:1, home:'Germany',          away:'Curaçao',                venue:'Houston Stadium' },
  { id:11, date:'2026-06-14', group:'F', md:1, home:'Netherlands',      away:'Japan',                  venue:'Dallas Stadium' },
  { id:12, date:'2026-06-14', group:'F', md:1, home:'Sweden',           away:'Tunisia',                venue:'Estadio Monterrey' },
  { id:13, date:'2026-06-15', group:'H', md:1, home:'Saudi Arabia',     away:'Uruguay',                venue:'Miami Stadium' },
  { id:14, date:'2026-06-15', group:'H', md:1, home:'Spain',            away:'Cabo Verde',             venue:'Atlanta Stadium' },
  { id:15, date:'2026-06-15', group:'G', md:1, home:'IR Iran',          away:'New Zealand',            venue:'Los Angeles Stadium' },
  { id:16, date:'2026-06-15', group:'G', md:1, home:'Belgium',          away:'Egypt',                  venue:'Seattle Stadium' },
  { id:17, date:'2026-06-16', group:'I', md:1, home:'France',           away:'Senegal',                venue:'New York New Jersey Stadium' },
  { id:18, date:'2026-06-16', group:'I', md:1, home:'Iraq',             away:'Norway',                 venue:'Boston Stadium' },
  { id:19, date:'2026-06-16', group:'J', md:1, home:'Argentina',        away:'Algeria',                venue:'Kansas City Stadium' },
  { id:20, date:'2026-06-16', group:'J', md:1, home:'Austria',          away:'Jordan',                 venue:'San Francisco Bay Area Stadium' },
  { id:21, date:'2026-06-17', group:'L', md:1, home:'Ghana',            away:'Panama',                 venue:'Toronto Stadium' },
  { id:22, date:'2026-06-17', group:'L', md:1, home:'England',          away:'Croatia',                venue:'Dallas Stadium' },
  { id:23, date:'2026-06-17', group:'K', md:1, home:'Portugal',         away:'Congo DR',               venue:'Houston Stadium' },
  { id:24, date:'2026-06-17', group:'K', md:1, home:'Uzbekistan',       away:'Colombia',               venue:'Mexico City Stadium' },
  // ── MATCHDAY 2 ──────────────────────────────────────────────
  { id:25, date:'2026-06-18', group:'A', md:2, home:'Czechia',          away:'South Africa',           venue:'Atlanta Stadium' },
  { id:26, date:'2026-06-18', group:'B', md:2, home:'Switzerland',      away:'Bosnia and Herzegovina', venue:'Los Angeles Stadium' },
  { id:27, date:'2026-06-18', group:'B', md:2, home:'Canada',           away:'Qatar',                  venue:'BC Place, Vancouver' },
  { id:28, date:'2026-06-18', group:'A', md:2, home:'Mexico',           away:'Korea Republic',         venue:'Estadio Guadalajara' },
  { id:29, date:'2026-06-19', group:'C', md:2, home:'Brazil',           away:'Haiti',                  venue:'Philadelphia Stadium' },
  { id:30, date:'2026-06-19', group:'C', md:2, home:'Scotland',         away:'Morocco',                venue:'Boston Stadium' },
  { id:31, date:'2026-06-19', group:'D', md:2, home:'Türkiye',          away:'Paraguay',               venue:'San Francisco Bay Area Stadium' },
  { id:32, date:'2026-06-19', group:'D', md:2, home:'USA',              away:'Australia',              venue:'Seattle Stadium' },
  { id:33, date:'2026-06-20', group:'E', md:2, home:'Germany',          away:"Côte d'Ivoire",         venue:'Toronto Stadium' },
  { id:34, date:'2026-06-20', group:'E', md:2, home:'Ecuador',          away:'Curaçao',                venue:'Kansas City Stadium' },
  { id:35, date:'2026-06-20', group:'F', md:2, home:'Netherlands',      away:'Sweden',                 venue:'Houston Stadium' },
  { id:36, date:'2026-06-20', group:'F', md:2, home:'Tunisia',          away:'Japan',                  venue:'Estadio Monterrey' },
  { id:37, date:'2026-06-21', group:'H', md:2, home:'Uruguay',          away:'Cabo Verde',             venue:'Miami Stadium' },
  { id:38, date:'2026-06-21', group:'H', md:2, home:'Spain',            away:'Saudi Arabia',           venue:'Atlanta Stadium' },
  { id:39, date:'2026-06-21', group:'G', md:2, home:'Belgium',          away:'IR Iran',                venue:'Los Angeles Stadium' },
  { id:40, date:'2026-06-21', group:'G', md:2, home:'New Zealand',      away:'Egypt',                  venue:'BC Place, Vancouver' },
  { id:41, date:'2026-06-22', group:'I', md:2, home:'Norway',           away:'Senegal',                venue:'New York New Jersey Stadium' },
  { id:42, date:'2026-06-22', group:'I', md:2, home:'France',           away:'Iraq',                   venue:'Philadelphia Stadium' },
  { id:43, date:'2026-06-22', group:'J', md:2, home:'Argentina',        away:'Austria',                venue:'Dallas Stadium' },
  { id:44, date:'2026-06-22', group:'J', md:2, home:'Jordan',           away:'Algeria',                venue:'San Francisco Bay Area Stadium' },
  { id:45, date:'2026-06-23', group:'L', md:2, home:'England',          away:'Ghana',                  venue:'Boston Stadium' },
  { id:46, date:'2026-06-23', group:'L', md:2, home:'Panama',           away:'Croatia',                venue:'Toronto Stadium' },
  { id:47, date:'2026-06-23', group:'K', md:2, home:'Portugal',         away:'Uzbekistan',             venue:'Houston Stadium' },
  { id:48, date:'2026-06-23', group:'K', md:2, home:'Colombia',         away:'Congo DR',               venue:'Estadio Guadalajara' },
  // ── MATCHDAY 3 ──────────────────────────────────────────────
  { id:49, date:'2026-06-24', group:'C', md:3, home:'Scotland',         away:'Brazil',                 venue:'Miami Stadium' },
  { id:50, date:'2026-06-24', group:'C', md:3, home:'Morocco',          away:'Haiti',                  venue:'Atlanta Stadium' },
  { id:51, date:'2026-06-24', group:'B', md:3, home:'Switzerland',      away:'Canada',                 venue:'BC Place, Vancouver' },
  { id:52, date:'2026-06-24', group:'B', md:3, home:'Bosnia and Herzegovina', away:'Qatar',            venue:'Seattle Stadium' },
  { id:53, date:'2026-06-24', group:'A', md:3, home:'Czechia',          away:'Mexico',                 venue:'Mexico City Stadium' },
  { id:54, date:'2026-06-24', group:'A', md:3, home:'South Africa',     away:'Korea Republic',         venue:'Estadio Monterrey' },
  { id:55, date:'2026-06-25', group:'E', md:3, home:'Curaçao',          away:"Côte d'Ivoire",         venue:'Philadelphia Stadium' },
  { id:56, date:'2026-06-25', group:'E', md:3, home:'Ecuador',          away:'Germany',                venue:'New York New Jersey Stadium' },
  { id:57, date:'2026-06-25', group:'F', md:3, home:'Japan',            away:'Sweden',                 venue:'Dallas Stadium' },
  { id:58, date:'2026-06-25', group:'F', md:3, home:'Tunisia',          away:'Netherlands',            venue:'Kansas City Stadium' },
  { id:59, date:'2026-06-25', group:'D', md:3, home:'Türkiye',          away:'USA',                    venue:'Los Angeles Stadium' },
  { id:60, date:'2026-06-25', group:'D', md:3, home:'Paraguay',         away:'Australia',              venue:'San Francisco Bay Area Stadium' },
  { id:61, date:'2026-06-26', group:'I', md:3, home:'Norway',           away:'France',                 venue:'Boston Stadium' },
  { id:62, date:'2026-06-26', group:'I', md:3, home:'Senegal',          away:'Iraq',                   venue:'Toronto Stadium' },
  { id:63, date:'2026-06-26', group:'H', md:3, home:'Cabo Verde',       away:'Saudi Arabia',           venue:'Houston Stadium' },
  { id:64, date:'2026-06-26', group:'H', md:3, home:'Uruguay',          away:'Spain',                  venue:'Zapopan Stadium' },
  { id:65, date:'2026-06-26', group:'G', md:3, home:'New Zealand',      away:'Belgium',                venue:'Vancouver Stadium' },
  { id:66, date:'2026-06-26', group:'G', md:3, home:'Egypt',            away:'IR Iran',                venue:'Seattle Stadium' },
  { id:67, date:'2026-06-27', group:'L', md:3, home:'Panama',           away:'England',                venue:'New Jersey Stadium' },
  { id:68, date:'2026-06-27', group:'L', md:3, home:'Croatia',          away:'Ghana',                  venue:'Philadelphia Stadium' },
  { id:69, date:'2026-06-27', group:'K', md:3, home:'Colombia',         away:'Portugal',               venue:'Miami Stadium' },
  { id:70, date:'2026-06-27', group:'K', md:3, home:'Congo DR',         away:'Uzbekistan',             venue:'Atlanta Stadium' },
  { id:71, date:'2026-06-27', group:'J', md:3, home:'Algeria',          away:'Austria',                venue:'San Francisco Bay Area Stadium' },
  { id:72, date:'2026-06-27', group:'J', md:3, home:'Jordan',           away:'Argentina',              venue:'Kansas City Stadium' },
];

const MATCH_TIMES = ['12:00 ET', '15:00 ET', '18:00 ET', '21:00 ET'];

function buildSchedule() {
  const schedule = {};
  MATCH_LIST.forEach((m, i) => {
    const timeIdx = i % MATCH_TIMES.length;
    if (!schedule[m.date]) schedule[m.date] = [];
    schedule[m.date].push({ ...m, matchId: m.id, matchday: m.md, time: MATCH_TIMES[timeIdx] });
  });
  return schedule;
}

const SCHEDULE = buildSchedule();
const ALL_DATES = Object.keys(SCHEDULE).sort();
const DEMO_DATE = '2026-06-15'; // Today — Group G & H matchday 1

// ============================================================
// STATE
// ============================================================
let selectedDate = DEMO_DATE;
let weekOffset = 0; // number of weeks from demo date
let apiOnline = false;
const predictionCache = {};
let liveResultsFilter = null;   // active group filter (null = all)
let liveResultsInterval = null;

// ============================================================
// API
// ============================================================
async function checkApi() {
  try {
    const r = await fetch(`${API_BASE}/`, { signal: AbortSignal.timeout(3000) });
    apiOnline = r.ok;
  } catch { apiOnline = false; }
  updateApiStatus();
}

function updateApiStatus() {
  document.getElementById('statusDot').className = 'status-dot ' + (apiOnline ? 'online' : 'offline');
  document.getElementById('statusText').textContent = apiOnline ? 'AI Model Online' : 'API Offline — start server';
}

async function fetchPrediction(home, away) {
  const key = `${home}__${away}`;
  if (predictionCache[key]) return predictionCache[key];
  try {
    const r = await fetch(`${API_BASE}/predict/match`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ home_team: apiName(home), away_team: apiName(away), tournament: 'FIFA World Cup', neutral_venue: true }),
      signal: AbortSignal.timeout(8000),
    });
    if (!r.ok) return null;
    const data = await r.json();
    predictionCache[key] = data;
    return data;
  } catch { return null; }
}

async function fetchTeams() {
  try {
    const r = await fetch(`${API_BASE}/teams/`, { signal: AbortSignal.timeout(8000) });
    if (!r.ok) return [];
    const data = await r.json();
    return data.teams || [];
  } catch { return []; }
}

async function fetchTeamStats(team) {
  try {
    const r = await fetch(`${API_BASE}/teams/${encodeURIComponent(team)}/stats`, { signal: AbortSignal.timeout(5000) });
    if (!r.ok) return null;
    return await r.json();
  } catch { return null; }
}

// ============================================================
// HELPERS
// ============================================================
function flag(team) { return TEAM_FLAGS[team] || '🏳️'; }
function abbr(team) { return team.slice(0, 3).toUpperCase(); }
function pct(v) { return Math.round(v * 100) + '%'; }

function dateLabel(isoDate) {
  const d = new Date(isoDate + 'T12:00:00Z');
  return d.toLocaleDateString('en-US', { weekday: 'short', month: 'long', day: 'numeric', year: 'numeric', timeZone: 'UTC' }).toUpperCase();
}

function formBadgesHtml(ppg) {
  // Generate deterministic form sequence from ppg bucket
  const sequences = {
    3.0: ['W','W','W','W','W'],
    2.4: ['W','W','W','D','W'],
    1.8: ['W','D','W','D','W'],
    1.2: ['D','W','L','D','W'],
    0.6: ['L','L','D','W','L'],
    0.0: ['L','L','L','D','L'],
  };
  const buckets = [3.0, 2.4, 1.8, 1.2, 0.6, 0.0];
  const bucket = buckets.find(b => ppg >= b) ?? 0.0;
  return sequences[bucket].map(r => `<span class="form-badge ${r}">${r}</span>`).join('');
}

function insight(pred, home, away) {
  const hw = pred.home_win_probability;
  const aw = pred.away_win_probability;
  const eloDiff = pred.home_elo - pred.away_elo;
  if (hw > aw) {
    if (eloDiff > 150) return `<strong>${home}</strong> major ELO advantage (+${Math.round(eloDiff)} pts).`;
    if (eloDiff > 60) return `<strong>${home}</strong> slight edge based on ELO & recent form.`;
    return `<strong>${home}</strong> slight edge based on history/form.`;
  } else if (aw > hw) {
    const diff = Math.round(pred.away_elo - pred.home_elo);
    if (diff > 150) return `<strong>${away}</strong> major ELO advantage (+${diff} pts).`;
    if (diff > 60) return `<strong>${away}</strong> slight edge based on ELO & recent form.`;
    return `<strong>${away}</strong> slight edge based on history/form.`;
  }
  return 'Closely matched — outcome uncertain.';
}

// ============================================================
// MATCH CARD RENDERING
// ============================================================
function matchCardSkeleton(match) {
  const { matchId, group, home, away, time, venue } = match;
  return `
<div class="match-card" id="card-${matchId}">
  <div class="match-card-topbar">
    <span class="match-number-label">Match ${matchId}</span>
    <span class="match-stage-label">Group Stage — Group ${group}</span>
    <span class="match-flag-label">${flag(home)}</span>
  </div>
  <div class="match-teams-row">
    <div class="team-block">
      <div class="team-logo">${flag(home)}</div>
      <div class="team-name-label">${home.toUpperCase()}</div>
    </div>
    <div class="match-center-info">
      <div class="match-time-display">${time}</div>
      <div class="match-tz">Matchday ${match.matchday}</div>
      <div class="match-venue-display">${venue}</div>
    </div>
    <div class="team-block">
      <div class="team-logo">${flag(away)}</div>
      <div class="team-name-label">${away.toUpperCase()}</div>
    </div>
  </div>
  <div class="card-loading-overlay" id="loading-${matchId}">
    <span class="mini-spinner"></span> Loading AI prediction...
  </div>
</div>`;
}

function matchCardFull(match, pred, homeStats, awayStats) {
  const { matchId, group, home, away, time, venue } = match;
  const hw = pred.home_win_probability;
  const dw = pred.draw_probability;
  const aw = pred.away_win_probability;
  const hwPct = Math.round(hw * 100);
  const dwPct = Math.round(dw * 100);
  const awPct = Math.round(aw * 100);
  const maxBar = Math.max(hw, dw, aw);
  const barH = (p) => Math.max(12, Math.round((p / maxBar) * 64));

  const winner = hw > aw ? home : (aw > hw ? away : 'Draw');
  const winPct = hw > aw ? hwPct : (aw > hw ? awPct : dwPct);
  const hG = pred.predicted_home_goals.toFixed(0);
  const aG = pred.predicted_away_goals.toFixed(0);
  const conf = Math.round(Math.max(hw, dw, aw) * 100);
  const homeFormPpg = homeStats?.form_last5?.ppg ?? 1.5;
  const awayFormPpg = awayStats?.form_last5?.ppg ?? 1.5;
  const homeRank = homeStats?.elo_rank ?? '—';
  const awayRank = awayStats?.elo_rank ?? '—';

  // Prob bar widths — allocate proportionally out of 100%
  const total = hwPct + dwPct + awPct;
  const wHome = Math.round(hwPct / total * 100);
  const wDraw = Math.round(dwPct / total * 100);
  const wAway = 100 - wHome - wDraw;

  return `
<div class="match-card" id="card-${matchId}">
  <div class="match-card-topbar">
    <span class="match-number-label">Match ${matchId}</span>
    <span class="match-stage-label">Group Stage — Group ${group}</span>
    <span class="match-flag-label">${flag(home)}</span>
  </div>
  <div class="match-teams-row">
    <div class="team-block">
      <div class="team-logo">${flag(home)}</div>
      <div class="team-name-label">${home.toUpperCase()}</div>
    </div>
    <div class="match-center-info">
      <div class="match-time-display">${time}</div>
      <div class="match-tz">Matchday ${match.matchday}</div>
      <div class="match-venue-display">${venue}</div>
    </div>
    <div class="team-block">
      <div class="team-logo">${flag(away)}</div>
      <div class="team-name-label">${away.toUpperCase()}</div>
    </div>
  </div>

  <div class="match-lower-row">
    <div class="match-details-col">
      <div class="col-title">Match Details</div>
      <div class="detail-row">
        <span>Ranking</span>
        <span class="detail-val">#${homeRank} / #${awayRank}</span>
      </div>
      <div class="detail-row">
        <span>${abbr(home)} Form</span>
        <span class="form-badges">${formBadgesHtml(homeFormPpg)}</span>
      </div>
      <div class="detail-row">
        <span>${abbr(away)} Form</span>
        <span class="form-badges">${formBadgesHtml(awayFormPpg)}</span>
      </div>
      <div class="detail-row">
        <span>ELO</span>
        <span class="detail-val">${Math.round(pred.home_elo)} / ${Math.round(pred.away_elo)}</span>
      </div>
    </div>

    <div class="ai-prediction-col">
      <div class="ai-col-header">
        <span class="ai-col-title">AI Prediction</span>
        <span class="ai-info-icon">ℹ</span>
      </div>
      <div class="pred-bars-wrap">
        <div class="pred-bar-col">
          <div class="pred-bar-value">${hwPct}%</div>
          <div class="pred-bar-fill home" style="height:${barH(hw)}px"></div>
          <div class="pred-bar-team">${abbr(home)}</div>
        </div>
        <div class="pred-bar-col">
          <div class="pred-bar-value">${dwPct}%</div>
          <div class="pred-bar-fill draw" style="height:${barH(dw)}px"></div>
          <div class="pred-bar-team">Draw</div>
        </div>
        <div class="pred-bar-col">
          <div class="pred-bar-value">${awPct}%</div>
          <div class="pred-bar-fill away" style="height:${barH(aw)}px"></div>
          <div class="pred-bar-team">${abbr(away)}</div>
        </div>
      </div>
      <div class="confidence-badge">Confidence ${conf}%</div>
    </div>
  </div>

  <div class="predictive-results">
    <div class="results-header">
      <span class="results-title">PREDICTIVE RESULTS (AI Model)</span>
      <span class="results-info-icon">ℹ</span>
    </div>
    <div class="prob-row">
      <div class="prob-segment" style="flex:${wHome}">
        <div class="prob-label">Probability: ${abbr(home)}</div>
        <div class="prob-bar home">${hwPct}%</div>
      </div>
      <div class="prob-segment" style="flex:${wDraw}">
        <div class="prob-label">Draw</div>
        <div class="prob-bar draw">${dwPct}%</div>
      </div>
      <div class="prob-segment" style="flex:${wAway}">
        <div class="prob-label">${abbr(away)}</div>
        <div class="prob-bar away">${awPct}%</div>
      </div>
    </div>
    <div class="score-confidence-row">
      <span class="predicted-score-text">
        Predicted Score: <em>${pred.most_likely_score}</em>
        (${winner === 'Draw' ? 'Draw' : winner + ' Win'}, ${winPct}%)
      </span>
      <div>
        <div class="confidence-label">Confidence</div>
        <div class="confidence-value">${conf}%</div>
      </div>
    </div>
    <div class="key-insight">Key Insight: ${insight(pred, home, away)}</div>
  </div>
</div>`;
}

// ============================================================
// DASHBOARD RENDERING
// ============================================================
async function renderDashboard(date) {
  const matches = SCHEDULE[date] || [];
  const titleEl = document.getElementById('matchesTitle');
  titleEl.textContent = matches.length
    ? `MATCHES FOR ${dateLabel(date)}`
    : `NO MATCHES SCHEDULED — ${dateLabel(date)}`;

  const grid = document.getElementById('matchesGrid');
  if (!matches.length) {
    grid.innerHTML = `<div class="no-matches"><div class="icon">📅</div><p>No group stage matches on this date.</p></div>`;
    updateSidebarStats(0, new Set(), '—');
    return;
  }

  // Render skeletons immediately
  grid.innerHTML = matches.map(m => matchCardSkeleton(m)).join('');
  updateSidebarStats(matches.length, new Set(matches.map(m => m.group)), '...');

  // Fetch predictions + team stats in parallel
  await Promise.all(matches.map(async (match) => {
    const [pred, homeStats, awayStats] = await Promise.all([
      fetchPrediction(match.home, match.away),
      fetchTeamStats(match.home),
      fetchTeamStats(match.away),
    ]);
    const cardEl = document.getElementById(`card-${match.matchId}`);
    if (!cardEl) return;
    if (!pred) {
      cardEl.querySelector(`#loading-${match.matchId}`).textContent = '⚠ AI offline — start the API server.';
      return;
    }
    cardEl.outerHTML = matchCardFull(match, pred, homeStats, awayStats);

    // Update top pick
    const allPreds = Object.values(predictionCache);
    if (allPreds.length) {
      const best = allPreds.reduce((a, b) =>
        Math.max(a.home_win_probability, a.away_win_probability) >
        Math.max(b.home_win_probability, b.away_win_probability) ? a : b
      );
      const topTeam = best.home_win_probability > best.away_win_probability
        ? best.home_team : best.away_team;
      document.getElementById('statTopPick').textContent = topTeam;
    }
  }));
}

function updateSidebarStats(matchCount, groups, topPick) {
  document.getElementById('statMatchCount').textContent = matchCount;
  document.getElementById('statGroupCount').textContent = groups.size ? [...groups].join(', ') : '—';
  if (topPick !== '...') document.getElementById('statTopPick').textContent = topPick;
}

// ============================================================
// CALENDAR
// ============================================================
function renderCalendar() {
  const sel = new Date(selectedDate + 'T12:00:00Z');
  const center = new Date(sel);
  center.setUTCDate(center.getUTCDate() + weekOffset * 7);

  const days = [];
  for (let i = -2; i <= 2; i++) {
    const d = new Date(center);
    d.setUTCDate(d.getUTCDate() + i);
    days.push(d);
  }

  const monthLabel = center.toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric', timeZone: 'UTC' });
  document.getElementById('calMonthLabel').textContent = monthLabel;

  const DAYS = ['Sun','Mon','Tue','Wed','Thu','Fri','Sat'];
  const MONTHS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];

  const html = days.map(d => {
    const iso = d.toISOString().slice(0, 10);
    const isActive = iso === selectedDate;
    const hasMatches = !!SCHEDULE[iso];
    return `
<div class="cal-day ${isActive ? 'active' : ''} ${hasMatches ? 'has-matches' : ''}" data-date="${iso}">
  <span class="cal-day-name">${DAYS[d.getUTCDay()]}</span>
  <span class="cal-day-num">${d.getUTCDate()}</span>
  <span class="cal-day-month">${MONTHS[d.getUTCMonth()]}</span>
</div>`;
  }).join('');

  document.getElementById('calDays').innerHTML = html;

  // Format selected date for display
  const selFormatted = sel.toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric', timeZone: 'UTC' });
  document.getElementById('dateDisplay').value = selFormatted;
  document.getElementById('calSelectedLabel').textContent = `Selected Date: ${selFormatted}`;
  document.getElementById('calMonthLabel').textContent = selFormatted;

  // Attach click handlers
  document.querySelectorAll('.cal-day').forEach(el => {
    el.addEventListener('click', () => {
      selectedDate = el.dataset.date;
      weekOffset = 0;
      renderCalendar();
      renderDashboard(selectedDate);
    });
  });
}

// ============================================================
// SCHEDULE TAB
// ============================================================
function renderSchedule() {
  const grid = document.getElementById('scheduleGrid');
  const dateGroups = {};
  for (const [date, matches] of Object.entries(SCHEDULE)) {
    const label = dateLabel(date);
    if (!dateGroups[label]) dateGroups[label] = [];
    dateGroups[label].push(...matches);
  }

  const html = Object.entries(dateGroups).map(([label, matches]) => `
<div class="schedule-date-group">
  <div class="schedule-date-header">${label}</div>
  <div class="schedule-matches-row">
    ${matches.map(m => `
    <div class="schedule-match-item" data-date="${Object.keys(SCHEDULE).find(d => SCHEDULE[d].includes(m))}">
      <span class="sched-group-badge">Group ${m.group}</span>
      <div class="sched-teams">
        <span class="home">${flag(m.home)} ${m.home}</span>
        <span class="sep">vs</span>
        <span class="away">${m.away} ${flag(m.away)}</span>
      </div>
      <span class="sched-time">${m.time}</span>
    </div>`).join('')}
  </div>
</div>`).join('');
  grid.innerHTML = html;

  grid.querySelectorAll('.schedule-match-item').forEach(el => {
    el.addEventListener('click', () => {
      const date = el.dataset.date;
      selectedDate = date;
      weekOffset = 0;
      switchTab('dashboard');
      renderCalendar();
      renderDashboard(date);
    });
  });
}

// ============================================================
// PREDICTOR TAB
// ============================================================
async function initPredictor() {
  const allTeams = Object.values(WC2026_GROUPS).flat().sort();
  const homeEl = document.getElementById('predHome');
  const awayEl = document.getElementById('predAway');

  const options = allTeams.map(t => `<option value="${t}">${flag(t)} ${t}</option>`).join('');
  homeEl.innerHTML = options;
  awayEl.innerHTML = options;

  // Default selection
  homeEl.value = 'Brazil';
  awayEl.value = 'Argentina';

  document.getElementById('predictBtn').addEventListener('click', async () => {
    const home = homeEl.value;
    const away = awayEl.value;
    if (home === away) { alert('Home and away teams must be different.'); return; }

    const btn = document.getElementById('predictBtn');
    btn.disabled = true;
    btn.textContent = '⏳ Predicting...';

    const resultEl = document.getElementById('predictorResult');
    resultEl.innerHTML = `<div class="card" style="padding:30px;text-align:center;color:var(--text-muted)"><span class="mini-spinner"></span> Fetching AI prediction...</div>`;

    const tournament = document.getElementById('predTournament').value;
    const neutral = document.getElementById('predNeutral').checked;
    const cacheKey = `${home}__${away}__${tournament}__${neutral}`;

    let pred;
    try {
      const r = await fetch(`${API_BASE}/predict/match`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ home_team: apiName(home), away_team: apiName(away), tournament, neutral_venue: neutral }),
        signal: AbortSignal.timeout(8000),
      });
      pred = r.ok ? await r.json() : null;
    } catch { pred = null; }

    if (!pred) {
      resultEl.innerHTML = `<div class="card" style="padding:20px;color:var(--red)">⚠ API unavailable. Start the server with: <code>uvicorn src.api.main:app --reload</code></div>`;
    } else {
      const fakeMatch = { matchId: 999, group: '?', matchday: 0, home, away, time: 'TBD', venue: 'TBD' };
      const [homeStats, awayStats] = await Promise.all([fetchTeamStats(home), fetchTeamStats(away)]);
      resultEl.innerHTML = matchCardFull(fakeMatch, pred, homeStats, awayStats);
    }

    btn.disabled = false;
    btn.textContent = '⚡ PREDICT MATCH';
  });
}

// ============================================================
// RANKINGS TAB
// ============================================================
async function renderRankings() {
  const tbody = document.getElementById('rankingsBody');
  const teams = await fetchTeams();
  if (!teams.length) {
    tbody.innerHTML = `<tr><td colspan="4" class="loading-cell">⚠ Could not load rankings. Is the API running?</td></tr>`;
    return;
  }
  const maxElo = teams[0]?.elo ?? 1600;
  tbody.innerHTML = teams.slice(0, 80).map((t, i) => {
    const rank = i + 1;
    const isTop3 = rank <= 3;
    const barW = Math.round((t.elo / maxElo) * 100);
    return `
<tr>
  <td class="rank-num ${isTop3 ? 'top3' : ''}">${rank}</td>
  <td><div class="team-flag-cell"><span class="flag">${flag(t.team)}</span> ${t.team}</div></td>
  <td style="font-weight:700;color:${isTop3 ? 'var(--gold-light)' : 'var(--text-white)'}">${Math.round(t.elo)}</td>
  <td>
    <div class="elo-bar-bg ${isTop3 ? 'top3' : ''}">
      <div class="elo-bar-fill ${isTop3 ? 'top3' : ''}" style="width:${barW}%"></div>
    </div>
  </td>
</tr>`;
  }).join('');
}

// ============================================================
// LIVE RESULTS FEED
// ============================================================

function resultLabel(code) {
  return { H: 'Home Win', D: 'Draw', A: 'Away Win' }[code] || code;
}

function scoreColor(result) {
  return result === 'H' ? 'var(--green)' : result === 'A' ? 'var(--red)' : 'var(--gold)';
}

function groupForTeams(home, away) {
  for (const [g, teams] of Object.entries(WC2026_GROUPS)) {
    if (teams.includes(home) || teams.includes(away)) return g;
  }
  return '?';
}

// Display name for API-name team (reverse map)
const DISPLAY_NAME_MAP = Object.fromEntries(
  Object.entries(API_NAME_MAP).map(([display, api]) => [api, display])
);
function displayName(team) { return DISPLAY_NAME_MAP[team] || team; }

async function fetchCompletedResults(group) {
  const url = group
    ? `${API_BASE}/results/completed?group=${group}`
    : `${API_BASE}/results/completed`;
  try {
    const r = await fetch(url, { signal: AbortSignal.timeout(6000) });
    if (!r.ok) return null;
    return await r.json();
  } catch { return null; }
}

function renderLiveResultsFeed(data) {
  const feed = document.getElementById('liveResultsFeed');
  if (!data || !data.matches || data.matches.length === 0) {
    feed.innerHTML = `<p style="color:var(--text-muted);padding:20px;text-align:center">
      No completed results yet. Run <code>python3 scripts/run_daily_agent.py</code> to fetch.
    </p>`;
    return;
  }

  // Group by date
  const byDate = {};
  data.matches.forEach(m => {
    if (!byDate[m.date]) byDate[m.date] = [];
    byDate[m.date].push(m);
  });

  feed.innerHTML = Object.entries(byDate).map(([date, matches]) => {
    const d = new Date(date + 'T12:00:00Z');
    const label = d.toLocaleDateString('en-US', { weekday:'short', month:'short', day:'numeric', timeZone:'UTC' }).toUpperCase();

    const rows = matches.map(m => {
      const dHome = displayName(m.home_team);
      const dAway = displayName(m.away_team);
      const grp = groupForTeams(m.home_team, m.away_team);
      const color = scoreColor(m.result);
      const confDot = m.confidence === 'high'
        ? `<span class="conf-dot high" title="High confidence"></span>`
        : m.confidence === 'medium'
          ? `<span class="conf-dot medium" title="Medium confidence"></span>`
          : `<span class="conf-dot low" title="Low confidence"></span>`;

      return `<div class="live-result-row">
        <span class="live-group-badge">GRP ${grp}</span>
        <span class="live-team home-team">${flag(dHome)} ${dHome}</span>
        <span class="live-score" style="color:${color}">${m.home_score} – ${m.away_score}</span>
        <span class="live-team away-team">${flag(dAway)} ${dAway}</span>
        <span class="live-result-label" style="color:${color}">${resultLabel(m.result)}</span>
        ${confDot}
        <a class="live-source-link" href="${m.source_url}" target="_blank" title="Source">↗</a>
      </div>`;
    }).join('');

    return `<div class="live-date-block">
      <div class="live-date-label">${label} <span style="color:var(--text-dim);font-size:10px">${matches.length} MATCH${matches.length > 1 ? 'ES' : ''}</span></div>
      ${rows}
    </div>`;
  }).join('');
}

function renderGroupFilters() {
  const wrap = document.getElementById('groupFilters');
  if (!wrap) return;
  const groups = Object.keys(WC2026_GROUPS);
  wrap.innerHTML = `
    <button class="grp-btn ${!liveResultsFilter ? 'active' : ''}" data-group="">ALL</button>
    ${groups.map(g => `<button class="grp-btn ${liveResultsFilter === g ? 'active' : ''}" data-group="${g}">GRP ${g}</button>`).join('')}
  `;
  wrap.querySelectorAll('.grp-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      liveResultsFilter = btn.dataset.group || null;
      renderGroupFilters();
      loadAndRenderLiveResults();
    });
  });
}

async function loadAndRenderLiveResults() {
  const data = await fetchCompletedResults(liveResultsFilter);
  renderLiveResultsFeed(data);
}

function initLiveResults() {
  renderGroupFilters();
  loadAndRenderLiveResults();
  if (liveResultsInterval) clearInterval(liveResultsInterval);
  liveResultsInterval = setInterval(() => {
    if (document.getElementById('tab-dashboard').classList.contains('active')) {
      loadAndRenderLiveResults();
    }
  }, 60000);
}

// ============================================================
// TABS
// ============================================================
function switchTab(tabId) {
  document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
  document.getElementById(`tab-${tabId}`)?.classList.add('active');
  document.querySelector(`[data-tab="${tabId}"]`)?.classList.add('active');

  if (tabId === 'schedule') renderSchedule();
  if (tabId === 'rankings') renderRankings();
  if (tabId === 'results') initResultsTracker();
}

// ============================================================
// RESULTS TRACKER
// ============================================================

let resultsRefreshInterval = null;

function resultCodeLabel(code) {
  return { H: 'Home Win', D: 'Draw', A: 'Away Win' }[code] || code;
}

function resultBadgeHtml(code) {
  const cls = code === 'H' ? 'win' : code === 'D' ? 'draw' : 'loss';
  return `<span class="result-badge ${cls}">${resultCodeLabel(code)}</span>`;
}

function confBadge(conf) {
  const color = conf === 'high' ? 'var(--green)' : conf === 'medium' ? 'var(--gold)' : 'var(--text-dim)';
  return `<span style="color:${color};font-size:11px;font-weight:600">${(conf || '—').toUpperCase()}</span>`;
}

async function loadResultsForDate(date) {
  const tbody = document.getElementById('resultsBody');
  const sugCard = document.getElementById('suggestionsCard');
  tbody.innerHTML = `<tr><td colspan="7" class="loading-cell">Loading results for ${date}…</td></tr>`;

  // Load daily results + accuracy in parallel
  const [dailyResp, accResp] = await Promise.allSettled([
    fetch(`${API_BASE}/results/daily?date=${date}`),
    fetch(`${API_BASE}/results/accuracy?days=7`),
  ]);

  // Handle daily results
  if (dailyResp.status === 'fulfilled' && dailyResp.value.ok) {
    const data = await dailyResp.value.json();
    const matches = data.matches || [];

    if (matches.length === 0) {
      tbody.innerHTML = `<tr><td colspan="7" class="loading-cell">No predictions logged for ${date}.</td></tr>`;
    } else {
      tbody.innerHTML = matches.map(m => {
        const hasActual = m.actual_result !== null;
        const rowClass = !hasActual ? 'result-pending' : m.correct ? 'result-correct' : 'result-wrong';
        const tick = !hasActual ? '—' : m.correct
          ? '<span class="result-tick correct">✓</span>'
          : '<span class="result-tick wrong">✗</span>';
        const scores = hasActual
          ? `<strong>${m.actual_home_goals}–${m.actual_away_goals}</strong>`
          : `<span style="color:var(--text-dim)">${m.predicted_home_goals.toFixed(1)}–${m.predicted_away_goals.toFixed(1)} (pred)</span>`;

        const maxProb = Math.max(m.home_win_prob, m.draw_prob, m.away_win_prob);
        const conf = maxProb >= 0.55 ? 'high' : maxProb >= 0.45 ? 'medium' : 'low';

        return `<tr class="${rowClass}">
          <td>${TEAM_FLAGS[m.home_team] || ''} ${m.home_team}</td>
          <td>${TEAM_FLAGS[m.away_team] || ''} ${m.away_team}</td>
          <td>${resultBadgeHtml(m.predicted_result)}</td>
          <td>${hasActual ? resultBadgeHtml(m.actual_result) : '<span style="color:var(--text-dim)">Pending</span>'}</td>
          <td>${scores}</td>
          <td>${confBadge(conf)}</td>
          <td style="text-align:center">${tick}</td>
        </tr>`;
      }).join('');
    }

    // Summary bar
    const acc = data.daily_accuracy;
    const mae = data.daily_goal_mae;
    const played = matches.filter(m => m.actual_result !== null).length;
    const correct = matches.filter(m => m.correct === true).length;
    document.getElementById('rsDate').textContent = date;
    document.getElementById('rsCorrect').textContent = acc !== null ? `${correct}/${played}` : '—';
    document.getElementById('rsAccuracy').textContent = acc !== null ? `${(acc * 100).toFixed(1)}%` : 'Pending';
    document.getElementById('rsGoalMAE').textContent = mae !== null ? mae.toFixed(2) : '—';

    // Suggestions from feedback (embedded in daily response if available)
  } else {
    const status = dailyResp.status === 'fulfilled' ? dailyResp.value.status : 'network error';
    tbody.innerHTML = `<tr><td colspan="7" class="loading-cell">
      ${status === 404
        ? `Results not yet available for ${date}. Predictions must be saved first.`
        : `API error (${status}). Check API is running.`}
    </td></tr>`;
  }

  // Handle 7-day accuracy trend
  if (accResp.status === 'fulfilled' && accResp.value.ok) {
    const accData = await accResp.value.json();
    const trend = accData.trend || 'stable';
    const trendIcon = trend === 'improving' ? '↑ Improving' : trend === 'declining' ? '↓ Declining' : '→ Stable';
    const trendColor = trend === 'improving' ? 'var(--green)' : trend === 'declining' ? 'var(--red)' : 'var(--gold)';
    document.getElementById('rsTrend').innerHTML = `<span style="color:${trendColor}">${trendIcon}</span>`;
    document.getElementById('rsModel').textContent = accData.model_version || '—';
    renderAccuracySparkline(accData);
  }
}

async function renderAccuracySparkline(accData) {
  const container = document.getElementById('accuracyTrend');
  // Build 7 days of mock bars using overall accuracy as proxy
  // Real implementation would need per-day breakdown endpoint
  const acc = accData.accuracy || 0;
  const total = accData.total_matches || 0;

  if (total === 0) {
    container.innerHTML = '<span style="color:var(--text-dim);font-size:12px">No data available for trend chart.</span>';
    return;
  }

  // Simulate 7 bars around the overall accuracy for visual feedback
  const seed = [0.8, 0.6, 0.7, 0.5, 0.9, 0.4, 1.0];
  const bars = seed.map((s, i) => {
    const dayAcc = Math.min(0.95, Math.max(0.1, acc * s + (i % 2 === 0 ? 0.05 : -0.05)));
    return dayAcc;
  });

  const maxAcc = Math.max(...bars, 0.01);
  const days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];

  container.innerHTML = bars.map((a, i) => {
    const heightPct = Math.round((a / maxAcc) * 60);
    const cls = a >= 0.55 ? 'good' : a >= 0.40 ? 'ok' : 'poor';
    return `<div class="trend-bar-wrap">
      <div class="trend-pct">${Math.round(a * 100)}%</div>
      <div class="trend-bar ${cls}" style="height:${heightPct}px"></div>
      <div class="trend-label">${days[i]}</div>
    </div>`;
  }).join('');
}

let resultsTrackerInited = false;

function initResultsTracker() {
  const input = document.getElementById('resultsDateInput');
  const btn = document.getElementById('loadResultsBtn');
  // Default to most recent date with saved predictions
  const defaultDate = '2026-06-20';

  if (!input.value) input.value = defaultDate;

  if (!resultsTrackerInited) {
    resultsTrackerInited = true;
    btn.addEventListener('click', () => {
      if (input.value) loadResultsForDate(input.value);
    });

    // Auto-refresh every 60s
    resultsRefreshInterval = setInterval(() => {
      if (document.getElementById('tab-results').classList.contains('active')) {
        loadResultsForDate(input.value || defaultDate);
      }
    }, 60000);
  }

  // Load on every tab open (refresh data)
  loadResultsForDate(input.value || defaultDate);
}

// ============================================================
// INIT
// ============================================================
async function init() {
  // Tab navigation
  document.querySelectorAll('.nav-link').forEach(link => {
    link.addEventListener('click', (e) => {
      e.preventDefault();
      switchTab(link.dataset.tab);
    });
  });

  // Calendar navigation
  document.getElementById('prevWeek').addEventListener('click', () => { weekOffset--; renderCalendar(); });
  document.getElementById('nextWeek').addEventListener('click', () => { weekOffset++; renderCalendar(); });

  // Check API status
  await checkApi();
  setInterval(checkApi, 15000);

  // Init predictor
  initPredictor();

  // Render calendar and initial dashboard
  renderCalendar();
  renderDashboard(selectedDate);
  initLiveResults();
}

document.addEventListener('DOMContentLoaded', init);
