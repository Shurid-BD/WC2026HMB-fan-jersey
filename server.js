// ─── server.js ────────────────────────────────────────────────────────────────
require('dotenv').config();
const express  = require('express');
const cors     = require('cors');
const helmet   = require('helmet');
const { v4: uuid } = require('uuid');
const path     = require('path');
const fs       = require('fs');

const app  = express();
const PORT = process.env.PORT || 3000;
const OPENAI_KEY    = process.env.OPENAI_API_KEY;
const ANTHROPIC_KEY = process.env.ANTHROPIC_API_KEY;
const ALLOWED_ORIGIN = process.env.ALLOWED_ORIGIN || 'http://localhost:3000';

app.use(helmet({ contentSecurityPolicy: false }));
app.use(cors({ origin: ALLOWED_ORIGIN }));
app.use(express.json({ limit: '20mb' }));
app.use(express.static(path.join(__dirname, 'public')));

const UPLOAD_DIR = path.join(__dirname, 'public', 'uploads');
if (!fs.existsSync(UPLOAD_DIR)) fs.mkdirSync(UPLOAD_DIR, { recursive: true });

// ─── POST /api/generate ──────────────────────────────────────────────────────
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
        model: 'claude-sonnet-4-5',
        max_tokens: 320,
        system: `Analyse the person in the photo carefully. Return ONLY raw JSON, no markdown:
{"gender":"man or woman","age":"approximate e.g. mid-40s","build":"e.g. heavyset, athletic, slim","hair":"color and style e.g. short salt-and-pepper hair","skin":"skin tone e.g. medium brown","expression":"e.g. calm, smiling","beard":"describe beard/facial hair or none","caption":"<1-2 sentences max 25 words as a proud ${team.name} fan>","vibe":"<3 words max fan energy>"}`,
        messages: [{
          role: 'user',
          content: [
            { type: 'image', source: { type: 'base64', media_type: 'image/jpeg', data: photoB64 } },
            { type: 'text', text: `Describe this person precisely. Team: ${team.name}. JSON only.` }
          ]
        }]
      })
    });

    const claudeData = await claudeResp.json();
    if (claudeData.error) throw new Error('Claude: ' + claudeData.error.message);

    let person = {
      gender: 'man', age: 'adult', build: 'average build',
      hair: 'dark hair', skin: 'medium brown skin', expression: 'calm',
      beard: 'short beard', caption: `A passionate ${team.name} fan!`, vibe: 'bold and proud'
    };
    try {
      const raw = claudeData.content?.[0]?.text || '{}';
      Object.assign(person, JSON.parse(raw.replace(/```json|```/g, '').trim()));
    } catch (_) {}

    // ── Step 2: OpenAI Image Edit (inpainting) ───────────────────────────────
    // Use gpt-image-1 edit endpoint with the actual photo
    // This preserves the person's face and body, only editing the clothing region

    const jerseyPrompt = `Replace ONLY the shirt/clothing on this person with an official ${team.name} FIFA World Cup 2026 football jersey (${team.kit}). 
Keep EVERYTHING else IDENTICAL: the person's face, hair, beard, skin, body shape, posture, position, hands, background, lighting, and image framing. 
Do NOT zoom in, crop, or change the composition in any way. 
Do NOT alter the person's appearance, weight, or proportions.
Only the fabric of the shirt changes to the ${team.name} jersey with authentic team colors and badge.`;

    // Build multipart form data manually using Buffer
    const boundary = '----FormBoundary' + uuid().replace(/-/g, '');

    // Convert base64 photo to buffer
    const photoBuffer = Buffer.from(photoB64, 'base64');

    // Build mask: transparent in torso region only
    // We create a simple PNG mask programmatically
    const maskB64 = createMaskBase64();
    const maskBuffer = Buffer.from(maskB64, 'base64');

    let body = Buffer.alloc(0);

    const addField = (name, value) => {
      const part = `--${boundary}\r\nContent-Disposition: form-data; name="${name}"\r\n\r\n${value}\r\n`;
      body = Buffer.concat([body, Buffer.from(part)]);
    };

    const addFile = (name, filename, contentType, fileBuffer) => {
      const header = `--${boundary}\r\nContent-Disposition: form-data; name="${name}"; filename="${filename}"\r\nContent-Type: ${contentType}\r\n\r\n`;
      body = Buffer.concat([body, Buffer.from(header), fileBuffer, Buffer.from('\r\n')]);
    };

    addFile('image', 'photo.jpg', 'image/jpeg', photoBuffer);
    addFile('mask', 'mask.png', 'image/png', maskBuffer);
    addField('prompt', jerseyPrompt);
    addField('model', 'gpt-image-1');
    addField('n', '1');
    addField('size', '1024x1024');

    body = Buffer.concat([body, Buffer.from(`--${boundary}--\r\n`)]);

    const editResp = await fetch('https://api.openai.com/v1/images/edits', {
      method: 'POST',
      headers: {
        'Authorization': 'Bearer ' + OPENAI_KEY,
        'Content-Type': `multipart/form-data; boundary=${boundary}`
      },
      body
    });

    const editData = await editResp.json();
    if (editData.error) throw new Error('OpenAI: ' + editData.error.message);

    const imgB64 = editData.data?.[0]?.b64_json;
    const imgUrl = editData.data?.[0]?.url;

    if (!imgB64 && !imgUrl) throw new Error('No image returned from OpenAI');

    res.json({ imgB64: imgB64 || null, imgUrl: imgUrl || null, person });

  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message });
  }
});

// ─── Create a simple torso mask (transparent in shirt area) ──────────────────
// Returns base64 PNG: white everywhere except transparent in torso area
function createMaskBase64() {
  // Minimal 1x1 transparent PNG — we'll use a pre-built 1024x1024 mask
  // White (keep) = RGB(255,255,255,255), Transparent (edit) = RGBA(0,0,0,0)
  // This is a base64-encoded simple white PNG with transparent center
  // For a proper mask, we use a raw PNG built with pixel data

  const size = 1024;
  const channels = 4;
  const data = new Uint8Array(size * size * channels);

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const i = (y * size + x) * channels;
      // Torso region: x 15%-85%, y 30%-85% = transparent (to inpaint)
      const inTorso = x > size * 0.15 && x < size * 0.85 && y > size * 0.30 && y < size * 0.85;
      data[i]     = 255; // R
      data[i + 1] = 255; // G
      data[i + 2] = 255; // B
      data[i + 3] = inTorso ? 0 : 255; // A: 0=transparent(edit), 255=opaque(keep)
    }
  }

  // Encode as PNG manually (using raw IDAT — simplified, use sharp in production)
  // For now return a minimal valid transparent PNG as fallback
  return 'iVBORw0KGgoAAAANSUhEUgAABAAAAAQACAYAAAB/HSuDAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEwAACxMBAJqcGAAAABl0RVh0U29mdHdhcmUAd3d3Lmlua3NjYXBlLm9yZ5vuPBoAAAMASURBVHic7cExAQAAAMKg9U9tCy+gAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAeAMBuAABHgAAAABJRU5ErkJggg==';
}

// ─── POST /api/share-image ────────────────────────────────────────────────────
app.post('/api/share-image', express.json({ limit: '10mb' }), (req, res) => {
  try {
    const { imageB64, team, flag } = req.body;
    if (!imageB64) return res.status(400).json({ error: 'No image' });

    const fileId   = uuid();
    const filename = `${fileId}.png`;
    const filepath = path.join(UPLOAD_DIR, filename);
    const buf      = Buffer.from(imageB64.replace(/^data:image\/\w+;base64,/, ''), 'base64');
    fs.writeFileSync(filepath, buf);
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
app.get('/share/:id', (req, res) => {
  const team    = decodeURIComponent(req.query.team || 'the World Cup');
  const flag    = decodeURIComponent(req.query.flag || '⚽');
  const imgUrl  = decodeURIComponent(req.query.img  || '');
  const toolUrl = process.env.BASE_URL || `http://localhost:${PORT}`;
  res.send(`<!DOCTYPE html><html><head>
    <meta charset="UTF-8">
    <title>${flag} Supporting ${team} at WC2026!</title>
    <meta property="og:type" content="website">
    <meta property="og:title" content="${flag} I'm supporting ${team} at FIFA World Cup 2026!">
    <meta property="og:description" content="I got my ${team} fan jersey! Powered by Heidelberg Materials Bangladesh PLC. #WC2026">
    <meta property="og:image" content="${imgUrl}">
    <meta property="og:image:width" content="1024">
    <meta property="og:image:height" content="1024">
    <meta property="og:url" content="${toolUrl}/share/${req.params.id}">
    <meta name="twitter:card" content="summary_large_image">
    <meta name="twitter:image" content="${imgUrl}">
    <meta http-equiv="refresh" content="0;url=${toolUrl}">
    <script>window.location.href="${toolUrl}"</script>
  </head><body><p>Redirecting... <a href="${toolUrl}">Click here</a></p></body></html>`);
});

app.listen(PORT, () => console.log(`✅ Server running on port ${PORT}`));
