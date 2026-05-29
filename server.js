require('dotenv').config();
const express  = require('express');
const cors     = require('cors');
const helmet   = require('helmet');
const { v4: uuid } = require('uuid');
const path     = require('path');
const fs       = require('fs');

const app  = express();
const PORT = process.env.PORT || 3000;
const OPENAI_KEY     = process.env.OPENAI_API_KEY;
const ANTHROPIC_KEY  = process.env.ANTHROPIC_API_KEY;
const ALLOWED_ORIGIN = process.env.ALLOWED_ORIGIN || 'http://localhost:3000';

app.use(helmet({ contentSecurityPolicy: false }));
app.use(cors({ origin: ALLOWED_ORIGIN }));
app.use(express.json({ limit: '20mb' }));
app.use(express.static(path.join(__dirname, 'public')));

const UPLOAD_DIR = path.join(__dirname, 'public', 'uploads');
if (!fs.existsSync(UPLOAD_DIR)) fs.mkdirSync(UPLOAD_DIR, { recursive: true });

// ─── GET /api/logo ────────────────────────────────────────────────────────────
app.get('/api/logo', async (req, res) => {
  try {
    const r = await fetch('https://upload.wikimedia.org/wikipedia/commons/thumb/4/4c/Heidelberg_Materials_logo.svg/1200px-Heidelberg_Materials_logo.svg.png');
    const buf = Buffer.from(await r.arrayBuffer());
    res.set('Content-Type', 'image/png');
    res.set('Cache-Control', 'public, max-age=86400');
    res.send(buf);
  } catch(e) { res.status(500).json({ error: e.message }); }
});

// ─── POST /api/generate ──────────────────────────────────────────────────────
app.post('/api/generate', async (req, res) => {
  try {
    const { photoB64, team } = req.body;
    if (!photoB64 || !team) return res.status(400).json({ error: 'Missing photo or team' });

    // ── Step 1: Claude Vision — describe person AND get face bounding box ────
    const claudeResp = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': ANTHROPIC_KEY,
        'anthropic-version': '2023-06-01'
      },
      body: JSON.stringify({
        model: 'claude-sonnet-4-5',
        max_tokens: 400,
        system: `Analyse the person in the photo carefully. Return ONLY raw JSON, no markdown.
Find the face bounding box as fractions of image dimensions (0.0 to 1.0).
Include some padding around the face (about 15% extra on each side).
Format:
{
  "gender": "man or woman",
  "age": "e.g. mid-40s",
  "build": "e.g. heavyset",
  "hair": "e.g. short grey hair",
  "skin": "e.g. medium brown",
  "expression": "e.g. calm",
  "beard": "describe or none",
  "caption": "<max 25 words as proud ${team.name} fan>",
  "vibe": "<3 words>",
  "face": {
    "x": 0.35,
    "y": 0.05,
    "w": 0.30,
    "h": 0.28
  }
}
face.x = left edge, face.y = top edge, face.w = width, face.h = height — all as fractions of image size.`,
        messages: [{
          role: 'user',
          content: [
            { type: 'image', source: { type: 'base64', media_type: 'image/jpeg', data: photoB64 } },
            { type: 'text', text: `Team: ${team.name}. Find face bounding box precisely. JSON only.` }
          ]
        }]
      })
    });

    const claudeData = await claudeResp.json();
    if (claudeData.error) throw new Error('Claude: ' + claudeData.error.message);

    let person = {
      gender: 'man', age: 'adult', build: 'average', hair: 'dark hair',
      skin: 'medium skin', expression: 'calm', beard: 'short beard',
      caption: `A passionate ${team.name} fan!`, vibe: 'bold and proud',
      face: { x: 0.25, y: 0.05, w: 0.50, h: 0.30 }
    };
    try {
      const parsed = JSON.parse((claudeData.content?.[0]?.text || '{}').replace(/```json|```/g, '').trim());
      Object.assign(person, parsed);
      if (!person.face) person.face = { x: 0.25, y: 0.05, w: 0.50, h: 0.30 };
    } catch(_) {}

    // Clamp and validate face values — face should always be in upper 50% of image
    const f = person.face;
    f.x = Math.max(0, Math.min(0.85, f.x));
    f.y = Math.max(0, Math.min(0.45, f.y)); // face top never below 45% of image
    f.w = Math.max(0.15, Math.min(0.7, f.w));
    f.h = Math.max(0.15, Math.min(0.45, f.h)); // face height never more than 45%
    // Ensure face doesn't go out of bounds
    if (f.x + f.w > 1) f.w = 1 - f.x;
    if (f.y + f.h > 1) f.h = 1 - f.y;
    console.log('Face after clamp:', JSON.stringify(f));

    // ── Step 2: Resize original photo to 1024x1024 (fit, pad with white) ────
    const { Jimp } = require('jimp');
    const IMGSIZE = 1024;
    const photoBuffer = Buffer.from(photoB64, 'base64');
    const origImg = await Jimp.read(photoBuffer);
    const origW = origImg.width, origH = origImg.height;

    const scale = Math.min(IMGSIZE / origW, IMGSIZE / origH);
    const newW = Math.round(origW * scale);
    const newH = Math.round(origH * scale);
    const padX = Math.floor((IMGSIZE - newW) / 2);
    const padY = Math.floor((IMGSIZE - newH) / 2);

    const resized = origImg.clone().resize({ w: newW, h: newH });
    const padded = new Jimp({ width: IMGSIZE, height: IMGSIZE, color: 0xFFFFFFFF });
    padded.composite(resized, padX, padY);
    const finalPhotoBuffer = await padded.getBuffer('image/png');

    // ── Step 3: Extract face region from ORIGINAL resized image ─────────────
    // Face fractions are relative to the ORIGINAL image dimensions
    // We need to map them to the padded 1024x1024 coordinate space
    const facePixX = Math.round(padX + f.x * newW);
    const facePixY = Math.round(padY + f.y * newH);
    const facePixW = Math.round(f.w * newW);
    const facePixH = Math.round(f.h * newH);

    // Clamp to image bounds
    const cropX = Math.max(0, facePixX);
    const cropY = Math.max(0, facePixY);
    const cropW = Math.min(IMGSIZE - cropX, Math.max(1, facePixW));
    const cropH = Math.min(IMGSIZE - cropY, Math.max(1, facePixH));

    console.log(`Face region: x=${cropX}, y=${cropY}, w=${cropW}, h=${cropH}`);
    console.log(`Face fractions from Claude: x=${f.x}, y=${f.y}, w=${f.w}, h=${f.h}`);

    const faceRegion = padded.clone().crop({ x: cropX, y: cropY, w: cropW, h: cropH });

    // ── Step 4: Build mask — transparent only over shirt region ──────────────
    const maskBuffer = await buildMaskPng(IMGSIZE);

    // ── Step 5: OpenAI image edit ─────────────────────────────────────────────
    const jerseyPrompt = `Replace ONLY the shirt/clothing with an official ${team.name} FIFA World Cup 2026 football jersey (${team.kit}). Keep EVERYTHING else IDENTICAL: face, hair, beard, skin tone, body shape, posture, hands, background, lighting, and image framing. Do NOT zoom, crop, or change composition. Only the fabric of the shirt changes.`;

    const boundary = 'Boundary' + Date.now().toString(16);
    let body = Buffer.alloc(0);

    const addFile = (name, filename, type, buf) => {
      const h = `--${boundary}\r\nContent-Disposition: form-data; name="${name}"; filename="${filename}"\r\nContent-Type: ${type}\r\n\r\n`;
      body = Buffer.concat([body, Buffer.from(h), buf, Buffer.from('\r\n')]);
    };
    const addField = (name, val) => {
      body = Buffer.concat([body, Buffer.from(`--${boundary}\r\nContent-Disposition: form-data; name="${name}"\r\n\r\n${val}\r\n`)]);
    };

    addFile('image', 'photo.png', 'image/png', finalPhotoBuffer);
    addFile('mask',  'mask.png',  'image/png', maskBuffer);
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
    if (!imgB64 && !imgUrl) throw new Error('No image returned');

    // ── Step 6: Paste original upper body (head+neck) back onto generated ────
    let resultBuffer;
    try {
      const genImgBuffer = imgB64
        ? Buffer.from(imgB64, 'base64')
        : Buffer.from(await (await fetch(imgUrl)).arrayBuffer());

      const genImg = await Jimp.read(genImgBuffer);

      // Take top 38% of padded original (everything above shirt mask)
      // This covers head, neck, shoulders — exactly what we want to preserve
      const headH = Math.round(IMGSIZE * 0.38);
      const headRegion = padded.clone().crop({ x: 0, y: 0, w: IMGSIZE, h: headH });

      // Composite head region onto generated image at same position (top)
      genImg.composite(headRegion, 0, 0);

      resultBuffer = await genImg.getBuffer('image/png');
      console.log(`Head paste: top ${headH}px of original preserved`);
    } catch(e) {
      console.error('Face paste failed:', e.message);
      resultBuffer = imgB64 ? Buffer.from(imgB64, 'base64') : null;
    }

    const finalB64 = resultBuffer ? resultBuffer.toString('base64') : imgB64;
    res.json({ imgB64: finalB64, person });

  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message });
  }
});

// ─── Build mask PNG — transparent only over shirt region ─────────────────────
async function buildMaskPng(size) {
  const { Jimp } = require('jimp');
  const img = new Jimp({ width: size, height: size });
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const inShirt = x > size*0.10 && x < size*0.90 && y > size*0.38 && y < size*0.72;
      img.setPixelColor(inShirt ? 0x00000000 : 0xFFFFFFFF, x, y);
    }
  }
  return await img.getBuffer('image/png');
}

// ─── POST /api/share-image ────────────────────────────────────────────────────
app.post('/api/share-image', express.json({ limit: '10mb' }), (req, res) => {
  try {
    const { imageB64, team, flag } = req.body;
    if (!imageB64) return res.status(400).json({ error: 'No image' });

    const fileId   = uuid();
    const filename = `${fileId}.png`;
    const filepath = path.join(UPLOAD_DIR, filename);
    const buf = Buffer.from(imageB64.replace(/^data:image\/\w+;base64,/, ''), 'base64');
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
    <meta property="og:description" content="Get your fan jersey photo — powered by Heidelberg Materials Bangladesh PLC. #WC2026">
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
