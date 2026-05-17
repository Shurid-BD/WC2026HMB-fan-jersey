// ─── server.js ────────────────────────────────────────────────────────────────
// WC2026 Fan Jersey Generator — Backend
// npm install express cors helmet multer uuid dotenv
// ─────────────────────────────────────────────────────────────────────────────

require('dotenv').config();
const express  = require('express');
const cors     = require('cors');
const helmet   = require('helmet');
const multer   = require('multer');
const { v4: uuid } = require('uuid');
const path     = require('path');
const fs       = require('fs');

const app  = express();
const PORT = process.env.PORT || 3000;
const OPENAI_KEY    = process.env.OPENAI_API_KEY;
const ANTHROPIC_KEY = process.env.ANTHROPIC_API_KEY;
const ALLOWED_ORIGIN = process.env.ALLOWED_ORIGIN || 'http://localhost:3000';

// ─── Middleware ───────────────────────────────────────────────────────────────
app.use(helmet({ contentSecurityPolicy: false }));
app.use(cors({ origin: ALLOWED_ORIGIN }));
app.use(express.json({ limit: '15mb' }));
app.use(express.static(path.join(__dirname, 'public'))); // serve frontend

// Uploads folder for share images
const UPLOAD_DIR = path.join(__dirname, 'public', 'uploads');
if (!fs.existsSync(UPLOAD_DIR)) fs.mkdirSync(UPLOAD_DIR, { recursive: true });

// ─── POST /api/generate ──────────────────────────────────────────────────────
// 1. Receives base64 photo + team data
// 2. Calls Claude Vision to describe the person
// 3. Calls DALL-E 3 to generate jersey image
// 4. Returns base64 result image + AI caption data
app.post('/api/generate', async (req, res) => {
  try {
    const { photoB64, team } = req.body;
    if (!photoB64 || !team) return res.status(400).json({ error: 'Missing photo or team' });

    // ── Step 1: Claude Vision — describe the person ──────────────────────────
    const claudeResp = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': ANTHROPIC_KEY,
        'anthropic-version': '2023-06-01'
      },
      body: JSON.stringify({
        model: 'claude-sonnet-4-20250514',
        max_tokens: 320,
        system: `Analyse the person in the photo. Return ONLY raw JSON, no markdown:
{"gender":"man or woman","age":"approximate e.g. mid-30s","build":"e.g. athletic, slim, heavyset","hair":"color and style e.g. short black hair","skin":"skin tone e.g. light brown","expression":"e.g. smiling warmly","caption":"<1-2 sentences max 25 words as a proud ${team.name} fan, mention features>","vibe":"<3 words max fan energy>"}`,
        messages: [{
          role: 'user',
          content: [
            { type: 'image', source: { type: 'base64', media_type: 'image/jpeg', data: photoB64 } },
            { type: 'text', text: `Describe for DALL-E prompt. Team: ${team.name}. JSON only.` }
          ]
        }]
      })
    });

    const claudeData = await claudeResp.json();
    if (claudeData.error) throw new Error('Claude: ' + claudeData.error.message);

    let person = {
      gender: 'person', age: 'adult', build: 'average build',
      hair: 'dark hair', skin: 'medium skin tone', expression: 'smiling',
      caption: `A passionate ${team.name} fan ready for the World Cup!`,
      vibe: 'bold and proud'
    };
    try {
      const raw = claudeData.content?.[0]?.text || '{}';
      Object.assign(person, JSON.parse(raw.replace(/```json|```/g, '').trim()));
    } catch (_) {}

    // ── Step 2: DALL-E 3 — generate jersey image ─────────────────────────────
    const dallePrompt = `Photorealistic portrait photo of a ${person.age} ${person.gender} with ${person.hair}, ${person.skin}, ${person.build}, ${person.expression} expression, wearing an official ${team.name} FIFA World Cup 2026 football jersey (${team.kit}). The jersey fits naturally on the body. Upper body shot, clean white background, soft studio lighting, high quality photograph, sharp focus.`;

    const dalleResp = await fetch('https://api.openai.com/v1/images/generations', {
      method: 'POST',
      headers: {
        'Authorization': 'Bearer ' + OPENAI_KEY,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        model: 'dall-e-3',
        prompt: dallePrompt,
        n: 1,
        size: '1024x1024',
        response_format: 'b64_json',
        quality: 'standard'
      })
    });

    const dalleData = await dalleResp.json();
    if (dalleData.error) throw new Error('DALL-E: ' + dalleData.error.message);
    const imgB64 = dalleData.data?.[0]?.b64_json;
    if (!imgB64) throw new Error('No image returned from DALL-E');

    res.json({ imgB64, person });

  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message });
  }
});

// ─── POST /api/share-image ────────────────────────────────────────────────────
// Saves generated fan card PNG for dynamic Facebook OG sharing
app.post('/api/share-image', express.json({ limit: '5mb' }), (req, res) => {
  try {
    const { imageB64, team, flag } = req.body;
    if (!imageB64) return res.status(400).json({ error: 'No image' });

    const fileId   = uuid();
    const filename = `${fileId}.png`;
    const filepath = path.join(UPLOAD_DIR, filename);
    const buf      = Buffer.from(imageB64.replace(/^data:image\/\w+;base64,/, ''), 'base64');
    fs.writeFileSync(filepath, buf);

    // Auto-delete after 24h
    setTimeout(() => fs.unlink(filepath, () => {}), 24 * 60 * 60 * 1000);

    const baseUrl  = process.env.BASE_URL || `http://localhost:${PORT}`;
    const imageUrl = `${baseUrl}/uploads/${filename}`;
    const shareUrl = `${baseUrl}/share/${fileId}?team=${encodeURIComponent(team)}&flag=${encodeURIComponent(flag)}&img=${encodeURIComponent(imageUrl)}`;

    res.json({ shareUrl });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// ─── GET /share/:id ───────────────────────────────────────────────────────────
// Dynamic OG page for Facebook share previews
app.get('/share/:id', (req, res) => {
  const team   = decodeURIComponent(req.query.team || 'the World Cup');
  const flag   = decodeURIComponent(req.query.flag || '⚽');
  const imgUrl = decodeURIComponent(req.query.img  || '');
  const toolUrl = process.env.BASE_URL || `http://localhost:${PORT}`;

  res.send(`<!DOCTYPE html><html><head>
    <meta charset="UTF-8">
    <title>${flag} Supporting ${team} at WC2026!</title>
    <meta property="og:type" content="website">
    <meta property="og:title" content="${flag} I'm supporting ${team} at FIFA World Cup 2026!">
    <meta property="og:description" content="I got my ${team} fan jersey photo! Get yours — powered by Heidelberg Materials. #WC2026">
    <meta property="og:image" content="${imgUrl}">
    <meta property="og:image:width" content="800">
    <meta property="og:image:height" content="420">
    <meta property="og:url" content="${toolUrl}/share/${req.params.id}">
    <meta name="twitter:card" content="summary_large_image">
    <meta name="twitter:image" content="${imgUrl}">
    <meta http-equiv="refresh" content="0;url=${toolUrl}">
    <script>window.location.href="${toolUrl}"</script>
  </head><body><p>Redirecting... <a href="${toolUrl}">Click here</a></p></body></html>`);
});

app.listen(PORT, () => console.log(`✅ Server running on port ${PORT}`));
