#!/usr/bin/env node

/**
 * sync-skills.js — Syncs marketing skills to Supabase knowledge base
 *
 * Parses all SKILL.md files + references, chunks by section,
 * embeds via text-embedding-3-small (1536d), upserts to knowledge table.
 *
 * Env vars:
 *   OPENAI_API_KEY        — for embeddings
 *   SUPABASE_URL          — project API URL (e.g. https://xxx.supabase.co)
 *   SUPABASE_SERVICE_KEY  — service role key (bypasses RLS)
 *
 * Usage:
 *   node tools/clis/sync-skills.js sync              # full sync
 *   node tools/clis/sync-skills.js sync --dry-run    # preview without writing
 *   node tools/clis/sync-skills.js status            # show what's in Supabase
 *   node tools/clis/sync-skills.js clean             # remove stale skill chunks
 */

const { createHash } = require('crypto')
const { readFileSync, readdirSync, existsSync, statSync } = require('fs')
const { join, relative } = require('path')

// ── Config ──────────────────────────────────────────────────────────────────

// Load from OpenClaw config.json if env vars aren't set
function loadOpenClawConfig() {
  const candidates = [
    process.env.OPENCLAW_CONFIG,
    join(__dirname, '../../skills/supabase-memory/config.json'), // if running from OpenClaw
    // Common local paths
    join(process.env.HOME || process.env.USERPROFILE || '', 'OneDrive/Documents/1. PROJECTS/Programming/OpenClaw workspace/skills/supabase-memory/config.json'),
  ].filter(Boolean)
  for (const p of candidates) {
    try {
      return JSON.parse(readFileSync(p, 'utf-8'))
    } catch { /* skip */ }
  }
  return {}
}

// Load from OpenClaw .env for OPENAI_API_KEY
function loadOpenClawEnv() {
  const home = process.env.USERPROFILE || process.env.HOME || ''
  const candidates = [
    join(__dirname, '../../.env'),
    join(home, 'OneDrive/Documents/1. PROJECTS/Programming/OpenClaw workspace/.env'),
  ]
  for (const p of candidates) {
    try {
      const content = readFileSync(p, 'utf-8')
      const vars = {}
      for (const line of content.split('\n')) {
        if (!line || line.startsWith('#')) continue
        const eq = line.indexOf('=')
        if (eq === -1) continue
        vars[line.slice(0, eq).trim()] = line.slice(eq + 1).trim()
      }
      return vars
    } catch { /* skip */ }
  }
  return {}
}

const _ocCfg = loadOpenClawConfig()
const _ocEnv = loadOpenClawEnv()

// Prefer .env file over system env (system may have stale key)
const OPENAI_API_KEY = _ocEnv.OPENAI_API_KEY || process.env.OPENAI_API_KEY
const SUPABASE_URL = process.env.SUPABASE_URL || _ocCfg.supabaseUrl
const SUPABASE_SERVICE_KEY = process.env.SUPABASE_SERVICE_KEY || _ocCfg.supabaseServiceKey
const EMBEDDING_MODEL = 'text-embedding-3-small' // 1536 dims — matches production
const MAX_CHUNK_CHARS = 3000
const MIN_CHUNK_CHARS = 200
const SOURCE_TYPE = 'skill'
const CONFIDENCE = 'high'
const OWNER = 'jarvis'
const BATCH_SIZE = 5 // keep small for Tier 1 rate limits

// ── Arg parsing ─────────────────────────────────────────────────────────────

function parseArgs(argv) {
  const result = { _: [] }
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i]
    if (arg.startsWith('--')) {
      const key = arg.slice(2)
      const next = argv[i + 1]
      if (next && !next.startsWith('--')) {
        result[key] = next
        i++
      } else {
        result[key] = true
      }
    } else {
      result._.push(arg)
    }
  }
  return result
}

const args = parseArgs(process.argv.slice(2))
const command = args._[0]

// ── Helpers ─────────────────────────────────────────────────────────────────

/** Deterministic UUID v5-style from a string key (skill:chunk_index) */
function stableId(key) {
  const hash = createHash('sha256').update(key).digest('hex')
  // Format as UUID: 8-4-4-4-12
  return [
    hash.slice(0, 8),
    hash.slice(8, 12),
    '5' + hash.slice(13, 16), // version 5
    ((parseInt(hash.slice(16, 18), 16) & 0x3f) | 0x80).toString(16).padStart(2, '0') + hash.slice(18, 20), // variant
    hash.slice(20, 32),
  ].join('-')
}

/** Parse YAML frontmatter from a SKILL.md file */
function parseFrontmatter(content) {
  const match = content.match(/^---\r?\n([\s\S]*?)\r?\n---\r?\n([\s\S]*)$/)
  if (!match) return { meta: {}, body: content }

  const yamlStr = match[1]
  const body = match[2]
  const meta = {}

  // Simple YAML parser for flat + one-level nested keys
  let currentKey = null
  for (const line of yamlStr.split('\n')) {
    const topMatch = line.match(/^(\w[\w-]*):\s*(.*)$/)
    const nestedMatch = line.match(/^\s+(\w[\w-]*):\s*(.*)$/)

    if (topMatch) {
      currentKey = topMatch[1]
      const val = topMatch[2].trim()
      if (val) {
        meta[currentKey] = val.replace(/^['"]|['"]$/g, '')
      } else {
        meta[currentKey] = {}
      }
    } else if (nestedMatch && currentKey && typeof meta[currentKey] === 'object') {
      meta[currentKey][nestedMatch[1]] = nestedMatch[2].trim().replace(/^['"]|['"]$/g, '')
    }
  }
  return { meta, body }
}

/** Split markdown into sections by ## headers, merge small ones */
function chunkBySection(body, maxChars = MAX_CHUNK_CHARS) {
  const lines = body.split('\n')
  const sections = []
  let current = { heading: '', lines: [] }

  for (const line of lines) {
    if (line.match(/^##\s+/)) {
      if (current.lines.length > 0) {
        sections.push(current)
      }
      current = { heading: line.replace(/^##\s+/, ''), lines: [line] }
    } else {
      current.lines.push(line)
    }
  }
  if (current.lines.length > 0) sections.push(current)

  // Merge small sections, split large ones
  const chunks = []
  let buffer = []
  let bufferLen = 0

  for (const section of sections) {
    const text = section.lines.join('\n').trim()
    if (!text) continue

    if (bufferLen + text.length > maxChars && bufferLen > 0) {
      chunks.push(buffer.join('\n\n'))
      buffer = []
      bufferLen = 0
    }

    // Split oversized sections
    if (text.length > maxChars) {
      if (buffer.length > 0) {
        chunks.push(buffer.join('\n\n'))
        buffer = []
        bufferLen = 0
      }
      // Split by paragraphs
      const paragraphs = text.split(/\n\n+/)
      let paraBuffer = []
      let paraLen = 0
      for (const para of paragraphs) {
        if (paraLen + para.length > maxChars && paraLen > 0) {
          chunks.push(paraBuffer.join('\n\n'))
          paraBuffer = []
          paraLen = 0
        }
        paraBuffer.push(para)
        paraLen += para.length
      }
      if (paraBuffer.length > 0) {
        buffer = paraBuffer
        bufferLen = paraLen
      }
    } else {
      buffer.push(text)
      bufferLen += text.length
    }
  }

  if (buffer.length > 0) {
    chunks.push(buffer.join('\n\n'))
  }

  // Filter out tiny chunks
  return chunks.filter(c => c.length >= MIN_CHUNK_CHARS)
}

/** Map skill names to category tags */
function categoryTags(skillName) {
  const map = {
    'seo-audit': ['seo', 'audit'],
    'ai-seo': ['seo', 'ai-content'],
    'programmatic-seo': ['seo', 'programmatic-seo'],
    'site-architecture': ['seo', 'websites'],
    'schema-markup': ['seo', 'structured-data'],
    'content-strategy': ['content', 'content-strategy'],
    'page-cro': ['cro', 'conversion', 'websites'],
    'signup-flow-cro': ['cro', 'conversion', 'growth'],
    'onboarding-cro': ['cro', 'conversion', 'growth'],
    'form-cro': ['cro', 'conversion', 'lead-gen'],
    'popup-cro': ['cro', 'conversion', 'lead-gen'],
    'paywall-upgrade-cro': ['cro', 'conversion', 'monetization'],
    'copywriting': ['copywriting', 'content'],
    'copy-editing': ['copywriting', 'content'],
    'cold-email': ['cold_outreach', 'email'],
    'email-sequence': ['email', 'automation'],
    'social-content': ['social-media', 'content'],
    'paid-ads': ['paid-ads', 'ads'],
    'ad-creative': ['ads', 'creative'],
    'analytics-tracking': ['analytics', 'automation'],
    'ab-test-setup': ['cro', 'analytics'],
    'free-tool-strategy': ['growth', 'lead-gen'],
    'churn-prevention': ['retention', 'growth'],
    'referral-program': ['referral', 'growth'],
    'pricing-strategy': ['monetization', 'positioning'],
    'revops': ['automation', 'pipeline'],
    'sales-enablement': ['content', 'pipeline'],
    'launch-strategy': ['growth', 'positioning'],
    'competitor-alternatives': ['seo', 'positioning'],
    'product-marketing-context': ['positioning', 'marketing'],
    'marketing-ideas': ['marketing', 'growth'],
    'marketing-psychology': ['marketing', 'copywriting'],
  }
  return map[skillName] || ['marketing']
}

// ── Skill discovery ─────────────────────────────────────────────────────────

function findRepoRoot() {
  // Walk up from script location to find skills/ dir
  let dir = __dirname
  for (let i = 0; i < 5; i++) {
    if (existsSync(join(dir, 'skills'))) return dir
    dir = join(dir, '..')
  }
  // Fallback: cwd
  if (existsSync(join(process.cwd(), 'skills'))) return process.cwd()
  console.error(JSON.stringify({ error: 'Cannot find skills/ directory' }))
  process.exit(1)
}

function discoverSkills(repoRoot) {
  const skillsDir = join(repoRoot, 'skills')
  const skills = []

  for (const entry of readdirSync(skillsDir)) {
    const skillPath = join(skillsDir, entry, 'SKILL.md')
    if (!existsSync(skillPath)) continue

    const raw = readFileSync(skillPath, 'utf-8')
    const { meta, body } = parseFrontmatter(raw)
    const skillName = meta.name || entry

    // Collect reference files
    const refs = []
    const refsDir = join(skillsDir, entry, 'references')
    if (existsSync(refsDir) && statSync(refsDir).isDirectory()) {
      for (const refFile of readdirSync(refsDir)) {
        if (!refFile.endsWith('.md')) continue
        const refPath = join(refsDir, refFile)
        refs.push({
          filename: refFile,
          content: readFileSync(refPath, 'utf-8'),
        })
      }
    }

    skills.push({
      name: skillName,
      dir: entry,
      meta,
      body,
      refs,
      path: relative(repoRoot, skillPath),
    })
  }

  return skills.sort((a, b) => a.name.localeCompare(b.name))
}

/** Build all chunks for a single skill */
function buildSkillChunks(skill) {
  const chunks = []
  const tags = ['skill', skill.name, ...categoryTags(skill.name)]
  const description = typeof skill.meta.description === 'string' ? skill.meta.description : ''
  const version = skill.meta.metadata?.version || '1.1.0'

  // Chunk the main SKILL.md
  const mainChunks = chunkBySection(skill.body)
  for (let i = 0; i < mainChunks.length; i++) {
    const chunkKey = `skill:${skill.name}:main:${i}`
    chunks.push({
      id: stableId(chunkKey),
      title: `${skill.name} — skill${i > 0 ? ` (part ${i + 1})` : ''}`,
      content: mainChunks[i],
      chunk_index: i,
      source_url: `skills/${skill.dir}/SKILL.md`,
      source_type: SOURCE_TYPE,
      tags,
      confidence: CONFIDENCE,
      owner: OWNER,
      ingested_by: 'sync-skills',
      metadata: {
        skill_name: skill.name,
        version,
        description,
        file: 'SKILL.md',
        chunk_key: chunkKey,
      },
    })
  }

  // Chunk each reference file
  for (const ref of skill.refs) {
    const refChunks = chunkBySection(ref.content)
    for (let i = 0; i < refChunks.length; i++) {
      const chunkKey = `skill:${skill.name}:ref:${ref.filename}:${i}`
      chunks.push({
        id: stableId(chunkKey),
        title: `${skill.name} — ${ref.filename.replace('.md', '')}${i > 0 ? ` (part ${i + 1})` : ''}`,
        content: refChunks[i],
        chunk_index: chunks.length, // global index within this skill
        source_url: `skills/${skill.dir}/references/${ref.filename}`,
        source_type: SOURCE_TYPE,
        tags: [...tags, ref.filename.replace('.md', '')],
        confidence: CONFIDENCE,
        owner: OWNER,
        ingested_by: 'sync-skills',
        metadata: {
          skill_name: skill.name,
          version,
          description,
          file: `references/${ref.filename}`,
          chunk_key: chunkKey,
        },
      })
    }
  }

  return chunks
}

// ── Embedding ───────────────────────────────────────────────────────────────

async function embedBatch(texts) {
  const res = await fetch('https://api.openai.com/v1/embeddings', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${OPENAI_API_KEY}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model: EMBEDDING_MODEL,
      input: texts,
    }),
  })

  if (!res.ok) {
    const err = await res.text()
    throw new Error(`OpenAI embedding failed (${res.status}): ${err}`)
  }

  const data = await res.json()
  return data.data.map(d => d.embedding)
}

// ── Supabase upsert ─────────────────────────────────────────────────────────

async function upsertChunks(chunks) {
  // Supabase REST API upsert via POST with Prefer: resolution=merge-duplicates
  const url = `${SUPABASE_URL}/rest/v1/knowledge`
  const batchSize = 50

  let upserted = 0
  for (let i = 0; i < chunks.length; i += batchSize) {
    const batch = chunks.slice(i, i + batchSize)
    const res = await fetch(url, {
      method: 'POST',
      headers: {
        'apikey': SUPABASE_SERVICE_KEY,
        'Authorization': `Bearer ${SUPABASE_SERVICE_KEY}`,
        'Content-Type': 'application/json',
        'Prefer': 'resolution=merge-duplicates',
      },
      body: JSON.stringify(batch.map(c => ({
        id: c.id,
        title: c.title,
        content: c.content,
        chunk_index: c.chunk_index,
        source_url: c.source_url,
        source_type: c.source_type,
        tags: c.tags,
        confidence: c.confidence,
        owner: c.owner,
        ingested_by: c.ingested_by,
        metadata: c.metadata,
        embedding: c.embedding ? `[${c.embedding.join(',')}]` : null,
        updated_at: new Date().toISOString(),
      }))),
    })

    if (!res.ok) {
      const err = await res.text()
      throw new Error(`Supabase upsert failed (${res.status}): ${err}`)
    }
    upserted += batch.length
    process.stderr.write(`  upserted ${upserted}/${chunks.length}\r`)
  }
  console.error('') // newline
  return upserted
}

async function deleteStaleChunks(validIds) {
  // Delete skill chunks that no longer exist
  const url = `${SUPABASE_URL}/rest/v1/knowledge?source_type=eq.skill&id=not.in.(${validIds.join(',')})`
  const res = await fetch(url, {
    method: 'DELETE',
    headers: {
      'apikey': SUPABASE_SERVICE_KEY,
      'Authorization': `Bearer ${SUPABASE_SERVICE_KEY}`,
      'Prefer': 'return=representation',
    },
  })

  if (!res.ok) {
    const err = await res.text()
    throw new Error(`Supabase delete failed (${res.status}): ${err}`)
  }

  const deleted = await res.json()
  return deleted.length
}

async function getExistingSkillChunks() {
  const url = `${SUPABASE_URL}/rest/v1/knowledge?source_type=eq.skill&select=id,title,metadata,updated_at`
  const res = await fetch(url, {
    headers: {
      'apikey': SUPABASE_SERVICE_KEY,
      'Authorization': `Bearer ${SUPABASE_SERVICE_KEY}`,
    },
  })
  if (!res.ok) {
    const err = await res.text()
    throw new Error(`Supabase query failed (${res.status}): ${err}`)
  }
  return res.json()
}

// ── Commands ────────────────────────────────────────────────────────────────

async function cmdSync() {
  if (!OPENAI_API_KEY) {
    console.error(JSON.stringify({ error: 'OPENAI_API_KEY required' }))
    process.exit(1)
  }
  if (!args['dry-run'] && (!SUPABASE_URL || !SUPABASE_SERVICE_KEY)) {
    console.error(JSON.stringify({ error: 'SUPABASE_URL and SUPABASE_SERVICE_KEY required' }))
    process.exit(1)
  }

  const repoRoot = findRepoRoot()
  const skills = discoverSkills(repoRoot)
  console.error(`Found ${skills.length} skills`)

  // Build all chunks
  let allChunks = []
  for (const skill of skills) {
    const chunks = buildSkillChunks(skill)
    allChunks.push(...chunks)
    console.error(`  ${skill.name}: ${chunks.length} chunks (${skill.refs.length} refs)`)
  }
  console.error(`Total: ${allChunks.length} chunks`)

  if (args['dry-run']) {
    const summary = {
      dry_run: true,
      skills_found: skills.length,
      total_chunks: allChunks.length,
      skills: skills.map(s => ({
        name: s.name,
        chunks: allChunks.filter(c => c.metadata.skill_name === s.name).length,
        refs: s.refs.length,
        ref_files: s.refs.map(r => r.filename),
      })),
      sample_chunk: {
        id: allChunks[0]?.id,
        title: allChunks[0]?.title,
        tags: allChunks[0]?.tags,
        content_length: allChunks[0]?.content.length,
        content_preview: allChunks[0]?.content.slice(0, 200) + '...',
      },
    }
    console.log(JSON.stringify(summary, null, 2))
    return
  }

  // Generate embeddings in batches
  console.error('Generating embeddings...')
  for (let i = 0; i < allChunks.length; i += BATCH_SIZE) {
    const batch = allChunks.slice(i, i + BATCH_SIZE)
    const texts = batch.map(c => c.content)
    const embeddings = await embedBatch(texts)
    for (let j = 0; j < batch.length; j++) {
      batch[j].embedding = embeddings[j]
    }
    process.stderr.write(`  embedded ${Math.min(i + BATCH_SIZE, allChunks.length)}/${allChunks.length}\r`)
    // Small delay between batches to respect rate limits
    if (i + BATCH_SIZE < allChunks.length) await new Promise(r => setTimeout(r, 500))
  }
  console.error('')

  // Upsert to Supabase
  console.error('Upserting to Supabase...')
  const upserted = await upsertChunks(allChunks)
  console.error(`Upserted ${upserted} chunks`)

  // Clean up stale chunks
  const validIds = allChunks.map(c => c.id)
  const deleted = await deleteStaleChunks(validIds)
  if (deleted > 0) {
    console.error(`Cleaned up ${deleted} stale chunks`)
  }

  console.log(JSON.stringify({
    success: true,
    skills_synced: skills.length,
    chunks_upserted: upserted,
    stale_deleted: deleted,
  }))
}

async function cmdStatus() {
  if (!SUPABASE_URL || !SUPABASE_SERVICE_KEY) {
    console.error(JSON.stringify({ error: 'SUPABASE_URL and SUPABASE_SERVICE_KEY required' }))
    process.exit(1)
  }

  const existing = await getExistingSkillChunks()

  // Group by skill name
  const bySkill = {}
  for (const row of existing) {
    const name = row.metadata?.skill_name || 'unknown'
    if (!bySkill[name]) bySkill[name] = { chunks: 0, updated_at: null }
    bySkill[name].chunks++
    if (!bySkill[name].updated_at || row.updated_at > bySkill[name].updated_at) {
      bySkill[name].updated_at = row.updated_at
    }
  }

  console.log(JSON.stringify({
    total_skill_chunks: existing.length,
    skills: Object.entries(bySkill)
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([name, data]) => ({ name, ...data })),
  }, null, 2))
}

async function cmdClean() {
  if (!SUPABASE_URL || !SUPABASE_SERVICE_KEY) {
    console.error(JSON.stringify({ error: 'SUPABASE_URL and SUPABASE_SERVICE_KEY required' }))
    process.exit(1)
  }

  if (args['dry-run']) {
    const existing = await getExistingSkillChunks()
    console.log(JSON.stringify({
      dry_run: true,
      would_delete: existing.length,
      chunks: existing.map(r => ({ id: r.id, title: r.title })),
    }, null, 2))
    return
  }

  const url = `${SUPABASE_URL}/rest/v1/knowledge?source_type=eq.skill`
  const res = await fetch(url, {
    method: 'DELETE',
    headers: {
      'apikey': SUPABASE_SERVICE_KEY,
      'Authorization': `Bearer ${SUPABASE_SERVICE_KEY}`,
      'Prefer': 'return=representation',
    },
  })

  if (!res.ok) {
    const err = await res.text()
    throw new Error(`Delete failed (${res.status}): ${err}`)
  }

  const deleted = await res.json()
  console.log(JSON.stringify({ deleted: deleted.length }))
}

// ── Main ────────────────────────────────────────────────────────────────────

const commands = { sync: cmdSync, status: cmdStatus, clean: cmdClean }

if (!command || !commands[command]) {
  console.error(`Usage: node sync-skills.js <command> [--dry-run]

Commands:
  sync      Parse skills, embed, upsert to Supabase knowledge table
  status    Show current skill chunks in Supabase
  clean     Remove all skill chunks from Supabase

Options:
  --dry-run   Preview changes without writing

Environment:
  OPENAI_API_KEY        Required for sync (embeddings)
  SUPABASE_URL          Project API URL
  SUPABASE_SERVICE_KEY  Service role key`)
  process.exit(1)
}

commands[command]().catch(err => {
  console.error(JSON.stringify({ error: err.message }))
  process.exit(1)
})
