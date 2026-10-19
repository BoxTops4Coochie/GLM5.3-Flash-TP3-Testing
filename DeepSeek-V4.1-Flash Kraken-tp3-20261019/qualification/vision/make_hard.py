"""Harder deterministic vision set (seed 20261008): 5 categories x 12."""
import json, math, random, pathlib
from PIL import Image, ImageDraw, ImageFont
D = pathlib.Path(__file__).resolve().parent / 'hard'; D.mkdir(exist_ok=True)
F = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'; FM = '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'
rnd = random.Random(20261008); ALNUM = 'ACDEFGHJKLMNPQRTUVWXY34679'; items = []
def save(img, kind, i, q, a):
    p = D / f'{kind}-{i:02d}.png'; img.save(p); items.append(dict(id=p.stem, kind=kind, file=p.name, question=q, answer=a))
for i in range(12):  # small low-contrast OCR, 12 chars
    code = '-'.join(''.join(rnd.choice(ALNUM) for _ in range(4)) for _ in range(3))
    img = Image.new('RGB', (420, 90), (235, 235, 230)); d = ImageDraw.Draw(img)
    g = rnd.randint(70, 130); d.text((rnd.randint(10, 60), rnd.randint(20, 45)), code, fill=(g, g, g), font=ImageFont.truetype(FM, rnd.randint(13, 18)))
    save(img, 'smallocr', i, 'What exact code is shown in the image? Reply with only the code.', code)
for i in range(12):  # dense counting
    n = rnd.randint(13, 28); img = Image.new('RGB', (640, 420), 'white'); d = ImageDraw.Draw(img); pts = []
    while len(pts) < n:
        x, y, r = rnd.randint(25, 615), rnd.randint(25, 395), rnd.randint(9, 15)
        if all(math.hypot(x - a, y - b) > r + c + 6 for a, b, c in pts): pts.append((x, y, r))
    for x, y, r in pts: d.ellipse((x - r, y - r, x + r, y + r), fill=(40, 90, 200))
    save(img, 'densecount', i, 'How many circles are in the image? Reply with only the number.', str(n))
for i in range(12):  # exact bar value with gridlines
    vals = [rnd.randint(1, 20) for _ in range(5)]; k = rnd.randrange(5); labels = 'ABCDE'
    img = Image.new('RGB', (600, 420), 'white'); d = ImageDraw.Draw(img); f = ImageFont.truetype(F, 16)
    for v in range(0, 21, 2):
        y = 360 - 16 * v; d.line((60, y, 580, y), fill=(210, 210, 210)); d.text((25, y - 9), str(v), fill='black', font=f)
    d.line((60, 360, 580, 360), fill='black', width=2); d.line((60, 40, 60, 360), fill='black', width=2)
    for j, (lab, v) in enumerate(zip(labels, vals)):
        x = 85 + 98 * j; d.rectangle((x, 360 - 16 * v, x + 55, 360), fill=(200, 90, 60)); d.text((x + 22, 366), lab, fill='black', font=f)
    save(img, 'chartvalue', i, f'Using the y-axis gridlines, what is the value of bar {labels[k]}? Reply with only the integer.', str(vals[k]))
SHADES = {'navy': (20, 30, 110), 'blue': (40, 90, 230), 'teal': (20, 130, 130), 'cyan': (60, 210, 230), 'green': (40, 160, 60),
          'olive': (120, 130, 30), 'lime': (160, 230, 50), 'maroon': (120, 20, 30), 'red': (225, 30, 35), 'magenta': (220, 40, 200),
          'purple': (110, 40, 160), 'gold': (230, 180, 30)}
for i in range(12):  # fine shades in a 4x3 grid
    names = rnd.sample(list(SHADES), 12); img = Image.new('RGB', (500, 380), (230, 230, 230)); d = ImageDraw.Draw(img)
    for idx, nm in enumerate(names):
        r, c = divmod(idx, 4); d.rectangle((10 + 122 * c, 10 + 122 * r, 120 + 122 * c, 120 + 122 * r), fill=SHADES[nm])
    rr, cc = rnd.randint(1, 3), rnd.randint(1, 4)
    opts = ', '.join(sorted(SHADES))
    save(img, 'shade', i, f'The image is a grid of 3 rows and 4 columns of colored squares. What color is the square in row {rr}, column {cc} '
         f'(rows from the top, columns from the left, starting at 1)? Answer with exactly one of: {opts}.', names[(rr - 1) * 4 + (cc - 1)])
WORDS = 'river stone maple ember quartz harbor violet summit lantern meadow copper falcon willow glacier orbit candle'.split()
for i in range(12):  # small dense text, line k of 10
    lines = [' '.join(rnd.choice(WORDS) for _ in range(5)) for _ in range(10)]; k = rnd.randint(1, 10)
    img = Image.new('RGB', (520, 300), 'white'); d = ImageDraw.Draw(img); f = ImageFont.truetype(F, 16)
    for j, l in enumerate(lines): d.text((12, 8 + 28 * j), l, fill='black', font=f)
    save(img, 'denselines', i, f'Read line number {k} (counting from the top, starting at 1). Reply with only the words of that line.', lines[k - 1])
json.dump(items, open(D / 'answers.json', 'w'), indent=1); print(len(items), 'items')
