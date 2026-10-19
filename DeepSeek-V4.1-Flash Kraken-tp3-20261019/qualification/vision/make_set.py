"""Deterministic synthetic vision test set (seed 20261007): 5 categories x 12 images."""
import json, math, random, pathlib
from PIL import Image, ImageDraw, ImageFont
D = pathlib.Path(__file__).resolve().parent / 'set'; D.mkdir(exist_ok=True)
F = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'; FB = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
rnd = random.Random(20261007)
ALNUM = 'ACDEFGHJKLMNPQRTUVWXY34679'
COLORS = {'red': (220, 30, 30), 'green': (30, 160, 50), 'blue': (30, 70, 220), 'yellow': (240, 210, 20),
          'orange': (245, 130, 20), 'purple': (140, 40, 170), 'black': (15, 15, 15), 'pink': (245, 120, 190), 'brown': (130, 80, 40)}
WORDS = 'river stone maple ember quartz harbor violet summit lantern meadow copper falcon willow glacier orbit candle'.split()
items = []
def save(img, kind, i, q, a):
    p = D / f'{kind}-{i:02d}.png'; img.save(p); items.append(dict(id=p.stem, kind=kind, file=p.name, question=q, answer=a))
for i in range(12):  # OCR codes
    code = ''.join(rnd.choice(ALNUM) for _ in range(4)) + '-' + ''.join(rnd.choice(ALNUM) for _ in range(4))
    img = Image.new('RGB', (520, 160), (250, 250, 245)); d = ImageDraw.Draw(img)
    d.text((rnd.randint(15, 60), rnd.randint(30, 60)), code, fill=rnd.choice([(0, 0, 0), (30, 30, 120), (90, 20, 20)]),
           font=ImageFont.truetype(rnd.choice([F, FB]), rnd.randint(40, 56)))
    save(img, 'ocr', i, 'What exact code is shown in the image? Reply with only the code.', code)
for i in range(12):  # multi-line reading
    lines = [' '.join(rnd.choice(WORDS) for _ in range(4)) for _ in range(5)]
    k = rnd.randint(1, 5)
    img = Image.new('RGB', (640, 300), 'white'); d = ImageDraw.Draw(img); f = ImageFont.truetype(F, 30)
    for j, l in enumerate(lines): d.text((20, 15 + 55 * j), l, fill='black', font=f)
    save(img, 'lines', i, f'Read line number {k} (counting from the top, starting at 1). Reply with only the words of that line.', lines[k - 1])
for i in range(12):  # counting circles
    n = rnd.randint(3, 12); img = Image.new('RGB', (600, 400), 'white'); d = ImageDraw.Draw(img); pts = []
    while len(pts) < n:
        x, y, r = rnd.randint(40, 560), rnd.randint(40, 360), rnd.randint(18, 30)
        if all(math.hypot(x - a, y - b) > r + c + 12 for a, b, c in pts): pts.append((x, y, r))
    for x, y, r in pts: d.ellipse((x - r, y - r, x + r, y + r), fill=COLORS[rnd.choice(list(COLORS))])
    save(img, 'count', i, 'How many circles are in the image? Reply with only the number.', str(n))
for i in range(12):  # color grid
    names = rnd.sample(list(COLORS), 9); img = Image.new('RGB', (390, 390), (200, 200, 200)); d = ImageDraw.Draw(img)
    for idx, nm in enumerate(names):
        r, c = divmod(idx, 3); d.rectangle((15 + 125 * c, 15 + 125 * r, 125 + 125 * c, 125 + 125 * r), fill=COLORS[nm])
    rr, cc = rnd.randint(1, 3), rnd.randint(1, 3)
    save(img, 'grid', i, f'The image is a 3x3 grid of colored squares. What color is the square in row {rr}, column {cc} '
         '(rows from the top, columns from the left, both starting at 1)? Reply with one color word.', names[(rr - 1) * 3 + (cc - 1)])
for i in range(12):  # bar chart
    vals = rnd.sample(range(3, 20), 5); labels = 'ABCDE'; img = Image.new('RGB', (560, 380), 'white'); d = ImageDraw.Draw(img)
    f = ImageFont.truetype(F, 22); d.line((50, 330, 540, 330), fill='black', width=2); d.line((50, 20, 50, 330), fill='black', width=2)
    for j, (lab, v) in enumerate(zip(labels, vals)):
        x = 80 + 92 * j; d.rectangle((x, 330 - 15 * v, x + 55, 330), fill=(60, 110, 200)); d.text((x + 18, 338), lab, fill='black', font=f)
    save(img, 'chart', i, 'Which bar (letter) is the tallest? Reply with only the letter.', labels[vals.index(max(vals))])
json.dump(items, open(D / 'answers.json', 'w'), indent=1); print(len(items), 'items')
