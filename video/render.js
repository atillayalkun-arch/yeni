const { chromium } = require('playwright');
const http = require('http'), fs = require('fs'), path = require('path'), { spawn } = require('child_process');
const ROOT = __dirname, mime = { '.html': 'text/html', '.json': 'application/json', '.png': 'image/png' };
const srv = http.createServer((q, r) => { const f = path.join(ROOT, decodeURIComponent(q.url.split('?')[0]).replace(/^\/$/, '/index.html'));
  fs.readFile(f, (e, d) => { if (e) { r.writeHead(404); r.end(); } else { r.writeHead(200, { 'content-type': mime[path.extname(f)] || 'application/octet-stream' }); r.end(d); } }); });
async function main() {
  const [mode, dir, arg, nsubArg, workersArg] = process.argv.slice(2);
  await new Promise(r => srv.listen(0, r)); const port = srv.address().port;
  const browser = await chromium.launch({ args: ['--disable-gpu', '--no-sandbox'] });
  fs.mkdirSync(dir, { recursive: true });
  const frames = mode === 'full' ? Array.from({ length: 450 }, (_, i) => i) : arg.split(',').map(s => Math.round(parseFloat(s) * 30));
  const nsub = parseInt(nsubArg || '10'), nw = parseInt(workersArg || '1');
  let next = 0, done = 0; const t0 = Date.now();
  async function worker() {
    const ctx = await browser.newContext({ viewport: { width: 1376, height: 768 } }); const page = await ctx.newPage();
    page.on('console', m => { if (m.type() === 'error') console.log('console:', m.text()); }); page.on('pageerror', e => console.log('pageerror:', e.message));
    await page.goto(`http://localhost:${port}/index.html?render`); await page.waitForFunction('window.__ok === true', null, { timeout: 120000 });
    while (true) { const i = next++; if (i >= frames.length) break; const n = frames[i];
      const b64 = await page.evaluate(([n, s]) => window.renderFrame(n, s), [n, nsub]);
      fs.writeFileSync(path.join(dir, String(n).padStart(4, '0') + '.jpg'), Buffer.from(b64, 'base64'));
      if (++done % 25 === 0) console.log(done, '/', frames.length, ((Date.now() - t0) / 1000).toFixed(0) + 's'); }
    await ctx.close();
  }
  await Promise.all(Array.from({ length: nw }, worker));
  await browser.close(); srv.close(); console.log('bitti', ((Date.now() - t0) / 1000).toFixed(0) + 's');
}
main().catch(e => { console.error(e); process.exit(1); });
