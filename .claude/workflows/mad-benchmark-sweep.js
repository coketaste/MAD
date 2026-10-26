export const meta = {
  name: 'mad-benchmark-sweep',
  description: 'Fan out a benchmark matrix across MAD models/tags, run (or plan) each with madengine, parse results, and synthesize a comparison table',
  phases: [
    { title: 'Resolve', detail: 'expand tags into concrete models via madengine discover' },
    { title: 'Run', detail: 'execute each benchmark one at a time (or plan without agents)' },
    { title: 'Synthesize', detail: 'parse results into a comparison table' },
  ],
}

// ARGS — accepts EITHER:
//   (a) a structured object: { tags: string[], nGpus?: string[], execute?: boolean,
//       additionalContext?: string, timeout?: number }
//   (b) a CLI-style string mirroring `/mad-benchmark`, e.g.
//       --tags pyt_vllm_qwen3-8b,bert --additional-context '{"gpu_vendor":"AMD",...}' --live-output
//
// Multiple tags: pass several after --tags and/or comma-separate them. The Resolve
// phase expands each tag via `madengine discover`, so a group tag fans out into one
// sweep cell PER MODEL, models matched by several tags run once, and tags matching
// nothing are reported before any Docker build. --additional-context is threaded
// verbatim into EVERY cell command (so HF tokens / docker_env_vars reach each run).
//
// EXECUTE semantics: runs for real by default. Pass --no-execute (or --plan, or
// execute:false) to only validate commands + resolve tags without touching GPUs.
// Even when executing, each cell first checks for AMD GPUs and self-skips if none.
//
// NOTE: there is intentionally no precision axis. In MAD, precision is NOT a
// runtime `madengine run` flag: for training it is fixed per-model via
// `training_precision` in models.json, and for inference it is baked into the
// pre-trained model/image. To compare precisions, list the distinct model tags.

// Bash-single-quote an arbitrary string so it survives embedding in a shell command.
// MAD allows Python-dict --additional-context (single quotes); a raw '${ctx}' wrap
// breaks on those quotes and can inject shell metacharacters.
function shellQuote(s) {
  return "'" + String(s).replace(/'/g, "'\\''") + "'"
}

// --- tokenizer: split a CLI string respecting single/double quotes ---
function tokenize(s) {
  const out = []
  let cur = '', q = null
  for (let i = 0; i < s.length; i++) {
    const c = s[i]
    if (q) {
      if (c === q) q = null
      else cur += c
    } else if (c === '"' || c === "'") {
      q = c
    } else if (c === ' ' || c === '\t' || c === '\n') {
      if (cur) { out.push(cur); cur = '' }
    } else {
      cur += c
    }
  }
  if (cur) out.push(cur)
  return out
}

// --- parse a CLI string into the structured cfg the sweep uses ---
function parseCli(s) {
  const toks = tokenize(s)
  const cfg = { tags: [], nGpus: [], execute: true }
  for (let i = 0; i < toks.length; i++) {
    const t = toks[i]
    const next = () => toks[++i]
    if (t === '--tags' || t === '-t') {
      // collect following non-flag tokens, comma-splitting each
      while (i + 1 < toks.length && !toks[i + 1].startsWith('-')) {
        next().split(',').forEach(x => { if (x) cfg.tags.push(x) })
      }
    } else if (t === '--additional-context' || t === '-c') {
      cfg.additionalContext = next()
    } else if (t === '--ngpus' || t === '--n-gpus') {
      next().split(',').forEach(x => { if (x) cfg.nGpus.push(x) })
    } else if (t === '--timeout') {
      cfg.timeout = next()
    } else if (t === '-o' || t === '--output') {
      cfg.output = next()
    } else if (t === '--no-execute' || t === '--plan' || t === '--dry-run') {
      cfg.execute = false
    } else if (t === '--execute') {
      cfg.execute = true
    }
    // --live-output and unknown flags are ignored (live-output is always added below)
  }
  return cfg
}

// --- normalize args (string | object | undefined) into one cfg ---
let cfg
if (typeof args === 'string') {
  cfg = parseCli(args)
} else if (args && typeof args === 'object') {
  cfg = { execute: true, ...args }
  if (typeof cfg.tags === 'string') cfg.tags = [cfg.tags]
} else {
  cfg = { tags: [], nGpus: [], execute: true }
}

const tags = (cfg.tags && cfg.tags.length) ? cfg.tags : ['bert']
const nGpus = (cfg.nGpus && cfg.nGpus.length) ? cfg.nGpus : [null]
const execute = cfg.execute !== false
const addlCtx = cfg.additionalContext || null
const timeout = cfg.timeout || null

// --- Resolve: expand each tag into the concrete models it selects ---
phase('Resolve')
const RESOLVE_SCHEMA = {
  type: 'object',
  required: ['resolved', 'unresolved'],
  properties: {
    resolved: {
      type: 'array',
      items: {
        type: 'object',
        required: ['tag', 'models'],
        properties: {
          tag: { type: 'string' },
          models: {
            type: 'array',
            items: {
              type: 'object',
              required: ['name'],
              properties: { name: { type: 'string' }, n_gpus: { type: 'string' } },
            },
          },
        },
      },
    },
    unresolved: {
      type: 'array',
      items: {
        type: 'object',
        required: ['tag', 'error'],
        properties: { tag: { type: 'string' }, error: { type: 'string' } },
      },
    },
  },
}

const resolution = await agent(
  `Resolve MAD benchmark tags into concrete models. Do NOT build or run anything.
Pre-flight: run \`bash .claude/skills/mad-common/preflight.sh\` and stop if it exits non-zero.
For EACH tag below, run it on its own (so one bad tag cannot fail the others), from the MAD repo root:
  madengine discover --json --tags <tag> 2>/dev/null
stdout is a JSON array of model cards; exit code 1 means no model matched.
Tags: ${JSON.stringify(tags)}
For each tag that matched, report every model's exact "name" (keep any scope prefix such as "dummy/dummy_multi") and its "n_gpus".
For each tag that failed, put it in unresolved with the error message. Copy names verbatim — do not guess or invent models.`,
  { label: 'resolve:discover', phase: 'Resolve', schema: RESOLVE_SCHEMA, effort: 'low' }
)

// Dedup models across tags; remember which tag(s) selected each one.
let models = []
let unresolved = []
if (resolution) {
  const byName = new Map()
  for (const r of resolution.resolved || []) {
    for (const m of r.models || []) {
      if (!byName.has(m.name)) byName.set(m.name, { name: m.name, modelNGpus: m.n_gpus || null, viaTags: [] })
      byName.get(m.name).viaTags.push(r.tag)
    }
  }
  models = [...byName.values()]
  unresolved = resolution.unresolved || []
} else {
  // Resolve agent died — fall back to treating each raw tag as one cell.
  log('Resolve agent returned nothing; falling back to one cell per raw tag (no dedup or validation).')
  models = tags.map(t => ({ name: t, modelNGpus: null, viaTags: [t] }))
}
if (unresolved.length)
  log(`Unresolved tag(s), will be reported as errors and NOT run: ${unresolved.map(u => u.tag).join(', ')}`)

// Build the matrix of {model, nGpus} cells. A full model name is a valid --tags value.
const matrix = []
for (const m of models)
  for (const g of nGpus)
    matrix.push({ tag: m.name, nGpus: g, viaTags: m.viaTags, modelNGpus: m.modelNGpus })

log(`Sweep matrix: ${matrix.length} cell(s) from ${models.length} model(s) resolved from ${tags.length} tag(s) [${tags.join(', ')}]. execute=${execute}${addlCtx ? ', additional-context threaded' : ''}`)

const CELL_SCHEMA = {
  type: 'object',
  required: ['cell', 'command', 'status'],
  properties: {
    cell: { type: 'string' },
    command: { type: 'string' },
    status: { type: 'string', enum: ['planned', 'ran', 'skipped', 'error'] },
    performance: { type: ['number', 'null'] },
    metric: { type: ['string', 'null'] },
    notes: { type: 'string' },
  },
}

function buildCell(cell) {
  const label = `${cell.tag}${cell.nGpus ? '/' + cell.nGpus + 'gpu' : ''}`
  // Each cell writes its OWN perf file so parallel runs never clobber a shared
  // perf.csv. A safe filename is derived from the label.
  const safe = label.replace(/[^A-Za-z0-9._-]/g, '_')
  const outFile = `perf_${safe}.csv`
  const flags = [
    '--tags ' + cell.tag,
    '--live-output',
    '-o ' + outFile,
  ]
  if (timeout) flags.push('--timeout ' + timeout)
  // Thread the user-supplied --additional-context verbatim into every cell. If a
  // per-cell n_gpus axis is set, try to merge it into the JSON; fall back to the
  // raw string + a separate context if merge isn't possible.
  let ctx = ''
  if (addlCtx) {
    let merged = addlCtx
    if (cell.nGpus) {
      try {
        const obj = JSON.parse(addlCtx)
        obj.n_gpus = cell.nGpus
        merged = JSON.stringify(obj)
      } catch (e) { /* leave raw; n_gpus axis ignored for this cell */ }
    }
    ctx = ` --additional-context ${shellQuote(merged)}`
  } else if (cell.nGpus) {
    ctx = ` --additional-context ${shellQuote(JSON.stringify({ n_gpus: String(cell.nGpus) }))}`
  }
  const cmd = `madengine run ${flags.join(' ')}${ctx}`
  return { label, outFile, cmd }
}

const cellMeta = c => ({ _label: c.label, _output: c.outFile, _viaTags: c.viaTags, _modelNGpus: c.modelNGpus })
const cells = matrix.map(cell => ({ ...cell, ...buildCell(cell) }))

phase('Run')
const results = []
if (!execute) {
  // Resolve already validated every tag via discover — planning needs no agents.
  for (const c of cells)
    results.push({ cell: c.label, command: c.cmd, status: 'planned', ...cellMeta(c) })
} else {
  // Run cells ONE AT A TIME: they share the same GPUs (and Docker daemon), so
  // concurrent runs would contend and skew each other's numbers.
  for (const c of cells) {
    const r = await agent(
      `MAD benchmark sweep cell "${c.label}".
Pre-flight: run \`bash .claude/skills/mad-common/preflight.sh\` and stop if it exits non-zero.
Command: ${c.cmd}
If AMD GPUs are present (check rocm-smi/amd-smi), RUN this command from the MAD repo root and parse results. This model may emit a "performance: <value> <unit>" stdout line OR (if it is a multiple_results model) write per-metric rows to its own CSV — in that case report the primary throughput metric. Results land in ${c.outFile}. If no GPUs, set status "skipped".
Return the cell label, the exact command, status, and performance/metric if available.`,
      { label: `bench:${c.label}`, phase: 'Run', schema: CELL_SCHEMA }
    )
    results.push(r
      ? { ...r, ...cellMeta(c) }
      : { cell: c.label, command: c.cmd, status: 'error', notes: 'cell agent returned no result', ...cellMeta(c) })
  }
}
// Unresolved tags never ran — surface them as error cells so they can't read as success.
for (const u of unresolved)
  results.push({ cell: u.tag, command: `madengine discover --tags ${u.tag}`, status: 'error', notes: `tag did not resolve: ${u.error}` })

phase('Synthesize')
const clean = results
const summary = await agent(
  `Synthesize a MAD benchmark sweep into a markdown comparison table.
Cells (JSON, each with its own _output perf file; _viaTags = the sweep tag(s) that selected the model): ${JSON.stringify(clean)}
Produce: (1) a table with columns Cell | Via tag(s) | Command | Status | Performance | Metric;
(2) a short headline (how many planned/ran/skipped/errored);
(3) explicitly flag any cell whose status is "error" or whose tag did not resolve —
do not let a failed/unresolved cell read as success;
(4) if any ran, the fastest and slowest cells (respecting that metric units differ);
(5) note that each cell wrote a separate perf_<cell>.csv (to avoid clobbering) and
that they can be concatenated for a combined report.
Do not invent numbers — only use provided data.`,
  { phase: 'Synthesize' }
)

return { execute, cells: clean, report: summary }
