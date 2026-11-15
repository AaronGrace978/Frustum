"""Build Frustum_loader_Whitepaper_v0.6.0.pdf from this script and whitepaper/assets.

Run from the repo root:  python whitepaper/build_whitepaper.py
Needs reportlab and pypdf. Evidence JSONs in the repo root are embedded as attachments.
"""
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, Preformatted,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parent.parent
WP = ROOT / 'whitepaper'
A = WP / 'assets'
OUT = WP / 'Frustum_loader_Whitepaper_v0.6.0.pdf'
VERSION = 'Version 0.6.0 | November 14, 2026'

FONT_DIRS = [Path('/usr/share/fonts/truetype/dejavu'), Path('C:/Windows/Fonts')]


def _font(name, files):
    for d in FONT_DIRS:
        for f in files:
            if (d / f).exists():
                pdfmetrics.registerFont(TTFont(name, str(d / f)))
                return
    raise FileNotFoundError(files)


_font('Body', ['DejaVuSans.ttf', 'arial.ttf'])
_font('Bold', ['DejaVuSans-Bold.ttf', 'arialbd.ttf'])
_font('Ital', ['DejaVuSans-Oblique.ttf', 'ariali.ttf'])
_font('Mono', ['DejaVuSansMono.ttf', 'consola.ttf'])
pdfmetrics.registerFontFamily('Body', normal='Body', bold='Bold', italic='Ital', boldItalic='Bold')

INK = colors.HexColor('#1b2a3a')
TEAL = colors.HexColor('#0f7c80')
SOFT = colors.HexColor('#d5ecee')
RULE = colors.HexColor('#c9d3dc')
MUTED = colors.HexColor('#6b7a89')

S = {
    'title': ParagraphStyle('title', fontName='Bold', fontSize=30, leading=36, textColor=INK, spaceAfter=10),
    'sub': ParagraphStyle('sub', fontName='Body', fontSize=16, leading=21, textColor=TEAL, spaceAfter=16),
    'h1': ParagraphStyle('h1', fontName='Bold', fontSize=22, leading=27, textColor=INK, spaceAfter=12),
    'h2': ParagraphStyle('h2', fontName='Bold', fontSize=12.5, leading=16, textColor=TEAL, spaceBefore=10, spaceAfter=5),
    'body': ParagraphStyle('body', fontName='Body', fontSize=9.4, leading=13, textColor=INK, spaceAfter=7, alignment=TA_LEFT),
    'small': ParagraphStyle('small', fontName='Body', fontSize=8.4, leading=12, textColor=MUTED, spaceAfter=6),
    'cell': ParagraphStyle('cell', fontName='Body', fontSize=8.6, leading=11.5, textColor=INK),
    'cellh': ParagraphStyle('cellh', fontName='Bold', fontSize=8.6, leading=11.5, textColor=INK),
    'code': ParagraphStyle('code', fontName='Mono', fontSize=8.2, leading=11, textColor=INK, spaceAfter=8),
    'cap': ParagraphStyle('cap', fontName='Ital', fontSize=8.4, leading=12, textColor=MUTED, spaceBefore=4, spaceAfter=10),
}

DINO = f'<img src="{A / "dino.png"}" width="12" height="12" valign="-2"/>'


def P(text, style='body'):
    return Paragraph(text, S[style])


def H1(text):
    return P(text, 'h1')


def H2(text):
    return P(text, 'h2')


def table(rows, widths=(2.55 * inch, 4.15 * inch), header=True):
    data = [[Paragraph(str(c), S['cellh' if header and i == 0 else 'cell']) for c in r]
            for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, hAlign='LEFT')
    style = [
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 4), ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 7), ('RIGHTPADDING', (0, 0), (-1, -1), 7),
        ('LINEBELOW', (0, 0), (-1, -1), 0.5, RULE),
    ]
    if header:
        style.append(('BACKGROUND', (0, 0), (-1, 0), SOFT))
    t.setStyle(TableStyle(style))
    return KeepTogether([t, Spacer(1, 8)])


def picture(path, width, caption=None):
    from PIL import Image as PILImage
    w, h = PILImage.open(path).size
    img = Image(str(path), width=width, height=width * h / w)
    img.hAlign = 'LEFT'
    parts = [img]
    if caption:
        parts.append(P(caption, 'cap'))
    return KeepTogether(parts)


def code(text):
    return Preformatted(text.strip('\n'), S['code'])


def on_page(canvas, doc):
    canvas.saveState()
    w, h = letter
    canvas.setFont('Body', 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(0.75 * inch, h - 0.55 * inch, 'FRUSTUM_LOADER / EXPERIMENTAL WHITEPAPER')
    canvas.setStrokeColor(TEAL)
    canvas.setLineWidth(1.6)
    canvas.line(0.75 * inch, h - 0.66 * inch, w - 0.75 * inch, h - 0.66 * inch)
    canvas.drawString(0.75 * inch, 0.5 * inch, f'{VERSION} | Live ActivatePrime benchmarks')
    canvas.drawRightString(w - 0.75 * inch, 0.5 * inch, str(doc.page))
    canvas.restoreState()


def story():
    s = []
    # ---------------------------------------------------------------- cover
    s += [Spacer(1, 6), P('Frustum_loader', 'title'),
          P('Streaming a 510 GB mixture-of-experts model from a USB SSD through a 16 GB consumer GPU', 'sub'),
          P(f'Technical whitepaper | {VERSION}'),
          P('Project concept, architecture and direction: <b>Aaron Grace</b>. '
            'Engineering contributors: <b>OpenAI</b> (ChatGPT and Codex, v0.1 to v0.4 implementation and analysis) and '
            '<b>Claude / Anthropic</b> (v0.5 I/O, placement and cache engineering). '
            f'Contributor: Dino Buddy {DINO}. The project also received assistance from DinoClaw, '
            'Aaron Grace\u2019s original agent runtime.'),
          P('The project was previously published as <i>Frusium_loader</i>. The name is now Frustum_loader: '
            'the working set is the slice of the model that is \u201cin view\u201d of the current token, '
            'the way a camera frustum bounds what a game engine streams and renders. '
            'Experimental engineering report; not peer reviewed.', 'small'),
          H2('What the prototype demonstrates'),
          P('Frustum_loader runs three different checkpoints on one NVIDIA RTX 5060 Ti 16 GB, with the weights on a '
            'Samsung T7 USB SSD, from a desktop chat app (ActivatePrime). The largest, DeepSeek V4.1 Flash, is a '
            'roughly 510 GB checkpoint with 40 layers and 384 routed experts per layer. Version 0.3.8 produced its '
            'first six tokens in 1,118 s. Version 0.5.1, with the drive on a USB 3.x link, produces about one token '
            'every 8 s. Version 0.6 adds fused decode kernels for all three backends and next-layer expert '
            'prediction for DeepSeek (Section 15); their GPU measurements are pending.'),
          table([
              ['Checkpoint', 'Live result in ActivatePrime (v0.5.1, T7 on USB-C)'],
              ['DeepSeek V4.1 Flash (~510 GB)', '24 tokens in 192.2 s on the first turn (~8.0 s/token); 24 tokens in 227.1 s on the second turn (~9.5 s/token)'],
              ['GLM-4.7-Flash Q4_K_M (30B MoE)', 'Cold load plus a 9-token reply in 23 s (262.1 s on a USB 2.0-speed link). A longer chat reply: 96 tokens in 104.2 s'],
              ['Phi-4 14B Q4_K_S', 'Cold load plus a 36-token reply in 53.8 s (~1.5 s/token)'],
          ]),
          PageBreak()]

    # ---------------------------------------------------------------- contents
    s += [H1('Contents'),
          table([
              ['Section', 'Topic'],
              ['1', 'Target checkpoint and constraints'],
              ['2', 'Storage and scheduling design'],
              ['3', 'What v0.5 changed'],
              ['4', 'Numerical implementation'],
              ['5', 'Component validation results'],
              ['6', 'Sequential text blocks'],
              ['7', 'First local reply'],
              ['8', 'GPU-resident output head'],
              ['9', 'Expert cache benchmark'],
              ['10', 'Dino Buddies chat demo'],
              ['11', 'Phi-4 on Frustum_loader'],
              ['12', 'GLM-4.7-Flash on Frustum_loader'],
              ['13', 'GLM placement and warm chat'],
              ['14', 'v0.5.1 live benchmarks in ActivatePrime'],
              ['15', 'v0.6: fused decode kernels and next-layer prediction'],
              ['16', 'Interpretation and limitations'],
              ['17', 'Reproduction and integration'],
              ['18', 'Contributors and references'],
              ['A', 'Run evidence (embedded attachments)'],
          ], widths=(0.8 * inch, 5.9 * inch)),
          picture(A / 'deepseek_progress.png', 5.0 * inch,
                  'Figure 1. DeepSeek V4.1 Flash seconds per generated token across versions. Grey bars are recorded '
                  'v0.3.8/v0.4 runs (Sections 8 to 10); teal bars are the v0.5.1 live ActivatePrime turns (Section 14). '
                  'Each bar is one recorded run; the operating-system file cache was not controlled.'),
          PageBreak()]

    # ---------------------------------------------------------------- 1
    s += [H1('1. Target checkpoint and constraints'),
          P('The source of truth is the local folder G:/Models/DeepSeek-V4.1-Flash. Configuration, index entries, '
            'safetensors headers and the supplied inference code were inspected. The folder\u2019s model name and '
            'publisher claims were not independently authenticated; this report describes these local files. [L1\u2013L3]'),
          table([
              ['Property', 'Observed value'],
              ['Architecture class', 'DeepseekV41ForCausalLM'],
              ['Text backbone', '40 layers; hidden dimension 5,120'],
              ['MoE configuration', '384 routed experts; top-6; one shared expert'],
              ['Expert intermediate dimension', '2,304'],
              ['Attention configuration', '64 heads; head dimension 512; window 128'],
              ['Checkpoint packaging', '48 shards; 96,085 indexed tensors'],
              ['Tensor payload bytes', '510,286,023,000 (510.286 decimal GB)'],
              ['Stored formats', 'BF16, F32, FP8 E4M3, E8M0, packed FP4 in I8'],
          ]),
          P('The local model card describes a 20-layer causal encoder followed by a 20-layer decoder, compressed sparse '
            'attention, Engram memory, vision input and DSpark. It reports rounded active-parameter figures of 8B for '
            'prefill and 16B for decode. These are model-card claims, not independently derived execution counts.'),
          H2('Expert-size accounting'),
          P('One three-projection expert contains 3 \u00d7 5,120 \u00d7 2,304 = 35,389,440 logical weight parameters. '
            'Its three FP4 matrices occupy 17,694,720 bytes; per-32 E8M0 scales add 1,105,920 bytes, for 18,800,640 '
            'bytes total. Six routed experts therefore require 112,803,840 bytes per layer if none are cached, about '
            '4.5 GB per token across 40 layers. This excludes the shared expert, router, attention and all other state.'),
          H2('Operating envelope'),
          P('The target is a 16 GB RTX 5060 Ti with the weights on a Samsung T7 USB-attached SSD. The whole checkpoint '
            'is about 32 times larger than the GPU and several times larger than host RAM, so the design question is '
            'which bytes live where, and how few of the rest must cross the USB link per token. Driver allocations and '
            'other processes are outside the PyTorch budget; CPU staging and the OS page cache also consume host RAM.'),
          PageBreak()]

    # ---------------------------------------------------------------- 2
    s += [H1('2. Storage and scheduling design'),
          H2('Catalog first; payload only on demand'),
          P('Catalog reads config.json, model.safetensors.index.json and each shard\u2019s length-prefixed JSON header. '
            'It validates index/header agreement, tensor extents and shard paths, then records each tensor\u2019s file, '
            'byte offset, shape, dtype and size. The initial scan read 10,685,312 header bytes and zero tensor payload '
            'bytes. A shard is a storage container, not a scheduling unit.'),
          H2('Bounded transfer path'),
          P('RangeReader serves bounded byte ranges from the checkpoint. Through v0.4 it mapped allocation-aligned '
            'read-only windows; v0.5 keeps a persistent handle per shard per thread and issues large positioned reads '
            'straight into the destination buffer (the mmap path remains available as FRUSTUM_READER=mmap). Pinned '
            'host buffers feed a CUDA copy stream while a separate stream runs GPU work; events guard every buffer and '
            'cache entry until the computation that uses it completes. [W2]'),
          table([
              ['Stage', 'Responsibility'],
              ['SSD / OS file cache', 'Supply requested tensor or row ranges'],
              ['RangeReader (4 threads)', 'Positioned reads on persistent handles'],
              ['Host-RAM tier (optional)', 'Keep recently evicted routed experts in system memory'],
              ['Pinned staging + copy stream', 'Move bounded chunks to the GPU and publish readiness events'],
              ['GPU tier', 'Resident dense weights, routers, head; cached routed experts'],
              ['CUDA compute stream', 'Wait for readiness, decode and execute'],
              ['Completion event', 'Permit safe buffer and cache reuse'],
          ]),
          H2('Prefetch without changing the answer'),
          P('After a router has selected its experts, the next selected expert is known exactly. The implementation '
            'dispatches current-expert computation and then prepares the next expert while protecting both cache '
            'entries. A miss must fetch the selected expert; substituting an available expert would change the model. '
            'Predicting a later layer\u2019s routing decision is not implemented.'),
          H2('Placement on a 16 GB card'),
          P('The v0.5 placement plan puts the always-used weights on the GPU first: dense and attention weights, '
            'routers, the BF16 output head and the shared experts. About 2 GiB is left for routed experts, and the '
            'host-RAM tier is sized automatically from free system memory (FRUSTUM_DEEPSEEK_HOST_GIB overrides it).'),
          H2('Eviction'),
          P('Consecutive tokens reuse roughly 57% of their routed experts. When the GPU tier is smaller than one '
            'token\u2019s working set, plain LRU evicts experts just before they are needed again, so v0.5 ranks '
            'entries by access frequency with exponential decay (half-life of 8 passes) and falls back to LRU when '
            'the tier is large enough (cache_policy.py).'),
          PageBreak()]

    # ---------------------------------------------------------------- 3
    s += [H1('3. What v0.5 changed'),
          P('The v0.4 decode profile showed where a token\u2019s 58 seconds went: 2.3 GB of routed experts arriving '
            'from the drive at a flat 37 MB/s, with 5.6 GB of dense weights and 0.16 GB of router weights re-read on '
            'every token as well. Model arithmetic was not the bottleneck, so v0.5 changes where bytes come from and '
            'leaves the arithmetic alone.'),
          table([
              ['Area', 'v0.4', 'v0.5'],
              ['Disk reads', 'New 8 MiB mmap window per chunk; shard reopened per tensor', 'Persistent handles; large positioned reads straight into the destination'],
              ['Dense weights, routers', '256 MiB / 32 MiB LRU, re-read every token', 'Resident on the GPU when they fit'],
              ['Output head', 'Streamed by default: 1.3 GB read per token', 'Resident by default'],
              ['Routed experts', 'One GPU LRU, one read at a time', 'GPU tier, optional host-RAM tier, 4 reader threads'],
              ['Eviction', 'LRU', 'Frequency-ranked with decay when the GPU tier is small; LRU otherwise'],
              ['Dense FP8 decode', 'Three full-size FP32 temporaries', 'One, same products'],
              ['Follow-up turns', 'Whole conversation re-prefilled', 'Processed prefix reused when token IDs match exactly'],
              ['Output', 'Printed at the end; greedy only', 'Streamed; optional temperature, top-k, top-p'],
              ['Launchers', 'Hard-coded interpreter path', 'FRUSTUM_PYTHON, local .venv, then the original path'],
          ], widths=(1.45 * inch, 2.55 * inch, 2.7 * inch)),
          P('run_chat.ps1 --legacy restores the v0.4 placement and the mmap reader for a same-day A/B comparison.'),
          H2('The drive link'),
          P('The single largest change between the v0.4 numbers and the v0.5.1 live numbers was not code. The T7 had '
            'been connected through a USB 2.0-speed link and delivered about 50 MB/s. Moved to a USB-C port running at '
            'USB 3.x speed, the same drive reads at roughly 500 to 520 MB/s. Because every cold expert crosses that '
            'link, the tenfold bandwidth difference shows up directly in time to first token and in DeepSeek decode '
            '(Figure 2, Section 14).'),
          PageBreak()]

    # ---------------------------------------------------------------- 4
    s += [H1('4. Numerical implementation'),
          H2('Packed weights and activation conversion'),
          P('Routed expert bytes contain two E2M1 values: the low nibble is the even column and the high nibble is the '
            'odd column. Magnitudes are {0, 0.5, 1, 1.5, 2, 3, 4, 6}, with a sign bit. One E8M0 scale applies to each '
            'group of 32 columns. Finite scale byte e represents 2^(e\u2212127); 255 is rejected. The format agrees with '
            'NVIDIA\u2019s documented MXFP4 packing. [W1]'),
          P('The correctness path decodes one matrix at a time to FP32 and performs ordinary matrix multiplication. '
            'Inputs are rounded to FP8 E4M3 with per-32 power-of-two scales following the supplied '
            'activation-quantization code, and projection outputs are rounded to BF16. This is not native FP4 Tensor '
            'Core execution. Dense and shared-expert weights use FP8 E4M3 matrices with per-32\u00d732 E8M0 block scales.'),
          H2('Routing and expert combination'),
          P('The text router computes sqrt(softplus(xW<super>T</super>)), adds the correction bias only for top-k '
            'selection, gathers the six unbiased scores, normalizes their sum and multiplies by the configured route '
            'scale of 1.5, so the six mixture weights sum to approximately 1.5, not 1. Image-routing bias is not '
            'exercised by the text-only path.'),
          P('For each selected expert, gate = W1x and up = W3x. The gate branch is clamped above at 10; the up branch '
            'is clamped to [\u221210, 10]. The hidden vector is SiLU(gate) \u00d7 up. Its routing coefficient is applied '
            'before BF16 rounding and W2, as in the supplied implementation. Expert outputs accumulate in FP32, the '
            'shared expert is added and the combined output is returned in BF16. [L3]'),
          H2('Sparse tables and residual mixing'),
          P('SparseRows fetches only selected rows from BF16 token embeddings or FP8/E8M0 Engram tables, caches decoded '
            'rows and checks indices and layouts. RMSNorm and mHC coefficient generation, Sinkhorn normalization, '
            'collapse and expansion are implemented separately. The residual mixing matrix is interpreted as '
            '[source, destination], matching the supplied block code.'),
          PageBreak()]

    # ---------------------------------------------------------------- 5
    s += [H1('5. Component validation results'),
          P('Environment: Windows, RTX 5060 Ti 16 GB, Python 3.13 and torch 2.11.0+cu128 in an isolated environment. '
            'No model weights were modified.'),
          table([
              ['Experiment', 'Observed result'],
              ['Single expert: layer 0 / expert 0', 'PASS; zero, normal and 8\u00d7-normal inputs'],
              ['Single-expert output', 'Final BF16 output exactly matched reference'],
              ['Single-expert I/O / reserved VRAM', '18.80064 MB / 0.158203 GiB'],
              ['MoE sublayers 0\u20133', 'PASS; two synthetic inputs per sublayer'],
              ['Real expert selection', '12 distinct routed experts per tested layer + shared'],
              ['MoE logical reads / evictions', '1,059.861504 MB / 39'],
              ['MoE peak allocated / reserved', '0.414290 / 0.525391 GiB'],
              ['MoE cache hits incl. prefetch', '48 / 52 requests (92.31%)'],
              ['MoE validation runtime', '75.1083 s, including CPU reference work'],
          ]),
          P('Every tested MoE sublayer reported relative L2 error 0 and maximum absolute final-output error 0. The '
            'sublayers were tested independently with synthetic BF16 inputs, not chained as complete transformer layers. [L4]'),
          H2('Other checks'),
          P('All 256 possible packed FP4 bytes were decoded against an arithmetic reference. Single-expert pre-rounding '
            'GEMM errors were at most 2.67\u00d710<super>\u22125</super>. Sparse-table testing covered first, interior, '
            'last and duplicate rows: 31,512 logical lookup bytes plus 42,016 independent reference-read bytes; warm '
            'lookup added no file reads. Residual primitives passed against FP64 NumPy calculations using real layer-0 '
            'parameters. [L5\u2013L7]'),
          H2('What exact agreement means'),
          P('The reference independently implements weight decoding and uses NumPy FP64 matrix products and routing '
            'arithmetic. It shares PyTorch FP8 activation conversion and SwiGLU logic with the tested path. Exact final '
            'BF16 agreement therefore supports these components but does not establish an independent end-to-end '
            'oracle, original TileLang kernel parity or correctness for all inputs.'),
          PageBreak()]

    # ---------------------------------------------------------------- 6
    s += [H1('6. Sequential text blocks'),
          P('The Layer 0 reproducibility audit passed across two independent processes and against the saved baseline. '
            'The sequential run also reproduced the saved Layer 0 output fingerprint.'),
          table([
              ['Layer', 'Checkpoint execution'],
              ['0', 'Sliding-window latent attention'],
              ['1', 'Sparse Engram lookup and gated residual update before sliding-window attention'],
              ['2', 'Sliding window plus 2:1 compressed KV and causal index selection'],
              ['3', 'Sliding window plus the compressed KV and selections published by Layer 2'],
          ], widths=(1.0 * inch, 5.7 * inch)),
          P('The runner carries the hyper-connection residual stream and its pre-mix coefficients between blocks. '
            'Layer 2 keeps incomplete compression groups across decode steps. The compressor weights are stored as BF16 '
            'and promoted to FP32 for the learned pooling projections.'),
          P('Using checkpoint token IDs 100, 200 and 300, all four MoE sublayers passed the NumPy routing and '
            'matrix-product references. Full prefill versus a two-token prefix and one cached decode differed by '
            '6.24e-5 relative L2 and 0.000977 maximum absolute error. Peak PyTorch memory was 2.254 GiB allocated and '
            '3.039 GiB reserved; the validation requested 4,767.51 MB of logical weight reads.'),
          H2('Reproducibility notice'),
          P('An earlier complete layer-0 test passed internal checks but selected different experts across two reported '
            'sessions using the same token IDs. The historical cause was not established. Version 0.2 added a '
            'stream-ordering hardening change and identity checks; the later audit above passed. This does not promote '
            'the earlier PASS results to a full-model correctness claim.'),
          PageBreak()]

    # ---------------------------------------------------------------- 7
    s += [H1('7. First local reply'),
          P('The text runner executed all 40 blocks and streamed the BF16 output head for greedy decoding. For the '
            'prompt \u201cSay hello.\u201d the six-token run returned \u201cHello!\u201d followed by a waving-hand emoji '
            'and \u201cHow can\u201d. It stopped at the configured token limit.'),
          picture(A / 'first_local_reply_frustum.png', 6.7 * inch,
                  'Figure 3. The first local generation run, visualized from chat_smoke_results.json. 40 layers, six '
                  'generated tokens, 18 min 38 s, 3.06 GiB peak reserved.'),
          P('The single run took 1,117.56 s with 2.276 GiB allocated and 3.057 GiB reserved at peak. The output head '
            'alone requested 7.943 GB of logical weight reads. The result established short-prompt text generation '
            'from the full checkpoint on this card.'),
          code('Prompt: Say hello.\nToken IDs: [19923, 3, 52780, 236, 1730, 588]'),
          PageBreak()]

    # ---------------------------------------------------------------- 8
    s += [H1('8. GPU-resident output head'),
          P('This experiment repeated the \u201cSay hello.\u201d prompt with a six-token limit. The output head was '
            'loaded once into GPU memory in BF16. Scoring retained FP32 arithmetic and the original 4,096-row chunks.'),
          table([
              ['Measurement', 'Original streamed / GPU-resident'],
              ['Generation wall time', '1117.56 / 1141.11 s'],
              ['Logical output-head reads', '7.943 / 1.324 GB'],
              ['Peak PyTorch allocated memory', '2.276 / 3.510 GiB'],
              ['Peak PyTorch reserved memory', '3.057 / 4.303 GiB'],
          ]),
          P('Generated token IDs matched exactly. The resident run reached its first token in 706.36 s, including '
            '32.82 s for normalization, loading and scoring. Subsequent tokens averaged 86.95 s each; output-head work '
            'averaged 0.033 s and requested no additional head bytes. Each mode has one recorded run. Resident head is '
            'the default from v0.5 on.'),
          H1('9. Expert cache benchmark'),
          P('Both runs used \u201cSay hello.\u201d, a three-token limit, a GPU-resident output head and synchronized '
            'per-layer decode profiling across all 40 layers.'),
          table([
              ['Measurement', '256 MiB / 6 GiB expert cache'],
              ['Time to first token', '721.32 s / 609.85 s'],
              ['First cached decode', '279.85 s / 208.83 s'],
              ['Second cached decode', '68.89 s / 58.29 s'],
              ['First decode expert reads', '5.929 GB / 5.745 GB'],
              ['Second decode expert reads', '5.929 GB / 2.275 GB'],
              ['Peak PyTorch reserved', '4.291 GiB / 11.066 GiB'],
          ]),
          P('Token IDs matched the original generation prefix. The second cached decode requested 3.654 GB fewer logical '
            'expert bytes with the larger cache. This profile, with the drive still on the slow link, is the one that '
            'motivated the v0.5 changes in Section 3.'),
          PageBreak()]

    # ---------------------------------------------------------------- 10-11
    s += [H1('10. Dino Buddies chat demo'),
          H2('Prompt'), P(f'Do you like Dino Buddies? {DINO}'),
          H2('Model reply (DeepSeek V4.1 Flash, v0.4)'),
          P('I don\'t have personal feelings, so I can\'t "like" things the way you do \u2014 but I can definitely'),
          P('Local greedy generation across 40 layers with a resident output head and a 6,144 MiB expert cache. The run '
            'generated 24 text tokens out of a 24-token limit in 2,705.31 s (about 113 s per token including prefill). '
            'Peak PyTorch-reserved memory was 11.127 GiB. The same 24-token budget took 192.2 s in v0.5.1 (Section 14).'),
          Spacer(1, 10),
          H1('11. Phi-4 on Frustum_loader'),
          P('Frustum_loader loads Microsoft Phi-4 14B Q4_K_S GGUF into GPU memory in packed form and runs all 40 layers '
            'in PyTorch on the RTX 5060 Ti.'),
          H2('Prompt'), P(f'Do you like Dino Buddies? {DINO} Reply in one short sentence.'),
          H2('Model reply'),
          P('I don\'t have personal preferences, but Dino Buddies is a popular children\'s toy line featuring dinosaur '
            'figures with interactive'),
          P('The smoke test stopped at the 24-token limit. The toy-line description is model output.', 'small'),
          table([
              ['Measurement', 'v0.4 smoke test / v0.5.1 ActivatePrime'],
              ['Packed weight load', '15.43 s / included below'],
              ['First token after loading', '18.07 s / included below'],
              ['Cold load plus reply', '\u2014 / 53.8 s for 36 tokens'],
              ['Mean cached decode', '1.43 s per token / ~1.5 s per token'],
              ['Peak PyTorch-reserved GPU memory', '8.48 GiB / not recorded'],
          ]),
          P('Phi-4 fits on the GPU, so its decode speed is set by per-token dequantization in PyTorch rather than the '
            'drive; the USB change did not alter it. Sampled dequantization and finite-logit checks passed.'),
          PageBreak()]

    # ---------------------------------------------------------------- 12-13
    s += [H1('12. GLM-4.7-Flash on Frustum_loader'),
          P('The GLM backend executes all 47 layers using Frustum_loader\u2019s bounded reader and PyTorch CUDA, loading '
            'routed quantized expert slices on demand.'),
          H2('Prompt'), P(f'Do you like Dino Buddies? {DINO} Reply in one short sentence.'),
          H2('Model reply'), P(f'Yes, I love Dino Buddies! {DINO}'),
          table([
              ['Measurement', 'Original smoke test (USB 2.0-speed link)'],
              ['First token, including on-demand reads', '58.70 s'],
              ['Mean cached decode', '1.46 s per step'],
              ['Peak PyTorch-reserved GPU memory', '9.88 GiB'],
              ['Logical weight payload reads', '12.45 GB'],
          ]),
          H1('13. GLM placement and warm chat'),
          P('The split policy reserves GPU space for attention, routers, shared experts and other always-used weights, '
            'with a separate cache for routed slices; the optional host tier retains evicted candidates in system RAM. '
            'ActivatePrime keeps one model worker warm, and GLM reuses its attention state when the formatted '
            'conversation starts with the exact cached token sequence. The app defaults to split placement.'),
          table([
              ['Policy (9 GiB GPU weight budget)', 'Warm first token, first / reverse-order run'],
              ['Unified GPU cache', '11.26 s / 11.20 s'],
              ['Split GPU cache', '9.80 s / 10.27 s'],
              ['Split GPU + host RAM', '17.78 s / not repeated'],
          ]),
          table([
              ['Split GPU follow-up', 'First / reverse-order run'],
              ['Full prompt rebuild', '11.72 s / 12.42 s'],
              ['Reuse 34 cached prompt tokens', '5.33 s / 5.28 s'],
          ]),
          P('Follow-up prompt: \u201cReply with only the word Dino.\u201d Model reply: \u201cDino\u201d. Both paths '
            'produced the same follow-up token IDs. The RAM tier cut warm logical file reads from 6.56 GB to 1.31 GB in '
            'the first comparison while increasing first-token latency. Split placement used about 9.99 GiB peak '
            'PyTorch-reserved memory.'),
          PageBreak()]

    # ---------------------------------------------------------------- 14
    s += [H1('14. v0.5.1 live benchmarks in ActivatePrime'),
          P('Hardware: NVIDIA RTX 5060 Ti 16 GB, Samsung T7 SSD on a USB-C port at USB 3.x speed, Windows. All runs '
            'were made from the ActivatePrime desktop app, which drives Frustum_loader through a warm Python worker. '
            'Times are the app\u2019s own wall-clock readout for the reply.'),
          table([
              ['Model', 'Run', 'Tokens', 'Wall time', 'Per token'],
              ['GLM-4.7-Flash', 'Cold load, USB 2.0-speed link', '9', '262.1 s', '\u2014'],
              ['GLM-4.7-Flash', 'Cold load, USB-C', '9', '23 s', '\u2014'],
              ['GLM-4.7-Flash', 'Chat reply (screenshot below)', '96', '104.2 s', '~1.09 s'],
              ['Phi-4 14B', 'Cold load, first reply', '36', '53.8 s', '~1.5 s'],
              ['DeepSeek V4.1 Flash', 'First turn (cold)', '24', '192.2 s', '~8.0 s'],
              ['DeepSeek V4.1 Flash', 'Second turn', '24', '227.1 s', '~9.5 s'],
          ], widths=(1.5 * inch, 2.15 * inch, 0.75 * inch, 1.0 * inch, 1.3 * inch)),
          picture(A / 'glm_usb_link.png', 5.9 * inch,
                  'Figure 2 (from Section 3). GLM-4.7-Flash cold load plus a 9-token reply, same drive and model, two USB links.'),
          H2('Reading the numbers'),
          P('Every timing in this paper is a single recorded run unless stated otherwise. Wall time includes model '
            'loading or prompt processing where noted. \u201cLogical reads\u201d count payload bytes requested from '
            'the checkpoint files, which the operating system may serve from its page cache; they are not physical '
            'SSD traffic. PyTorch \u201creserved\u201d memory is the caching allocator\u2019s footprint and excludes '
            'driver and other-process allocations.'),
          P('For DeepSeek, the earlier recorded best was 58.29 s for a single cached decode step (Section 9); the live '
            'figures are about seven times faster per token and include prompt processing. The second turn carries the '
            'longer conversation, so its wall time includes more context work than the first.'),
          PageBreak(),
          picture(A / 'activateprime_glm_chat.png', 4.9 * inch,
                  'Figure 4. ActivatePrime running GLM-4.7-Flash (30B MoE) locally through Frustum_loader on the RTX '
                  '5060 Ti 16 GB: 96 tokens in 104.2 s. The \u201cDino Buddy\u201d toy and its tag are the model\u2019s '
                  'own role-play.'),
          picture(A / 'activateprime_deepseek_chat.png', 4.9 * inch,
                  'Figure 5. DeepSeek V4.1 Flash, the full ~510 GB checkpoint, chatting in ActivatePrime on the same '
                  'card. Captured before the rename, so the app still shows the old Frusium label.'),
          PageBreak()]

    # ---------------------------------------------------------------- 15
    s += [H1('15. v0.6: fused decode kernels and next-layer prediction'),
          P('With the drive on a USB 3.x link, a large share of each token moved onto the GPU, and most of that '
            'work was not arithmetic. Every backend decoded each packed weight matrix into a full FP32 copy on every '
            'token, then multiplied a single row of activations by it. For a decode step that means writing and '
            're-reading several bytes per weight to use half a byte of packed data. Phi-4 shows it most clearly: '
            'its weights are fully resident, so its 1.4 to 1.5 s per token does not come from the drive.'),
          H2('Fused decode kernels'),
          P('For inputs of up to eight activation rows, one CUDA kernel reads the packed rows once, decodes them in '
            'registers and accumulates the dot products; one warp computes one output row and a warp shuffle '
            'finishes the sum. No decoded copy of the matrix is written. The source is compiled when the model '
            'loads, using NVRTC from the PyTorch CUDA wheel, and launched through the CUDA driver on PyTorch’s '
            'current stream, so nothing new is installed. Prefill keeps the reference path, where cuBLAS is the '
            'better tool.'),
          table([
              ['Format', 'Used by', 'Packed bytes per weight'],
              ['GGUF Q4_K / Q5_K / Q6_K / Q8_0', 'Phi-4, GLM-4.7-Flash', '0.56 / 0.69 / 0.82 / 1.06'],
              ['FP8 E4M3, E8M0 per 32×32 block', 'DeepSeek dense and shared experts', '≈1.0'],
              ['FP4 E2M1 packed, E8M0 per 32 columns', 'DeepSeek routed experts', '0.53'],
          ], widths=(2.5 * inch, 2.4 * inch, 1.8 * inch)),
          H2('Same numbers, different summation order'),
          P('Each weight is decoded with the same operations, in the same order, as the PyTorch reference '
            'decoders, with fused multiply-add disabled, and rounded to FP16 wherever the reference rounds. Decoded '
            'values are therefore bit-identical; only the order of the final summation differs from cuBLAS. At run '
            'time the first call for every matrix shape is also computed on the reference path and compared; any '
            'mismatch, compile failure or launch error switches the fused path off for the process and the reference '
            'path continues. FRUSTUM_FUSED=0 turns it off from the start.'),
          table([
              ['Check without a GPU (tools/fused_host_check.py)', 'Result'],
              ['Same CUDA source compiled with g++; 32 threads per warp so shuffles behave as on hardware', 'Builds and runs'],
              ['One-hot activations sweeping every column, all six formats, FP32 and FP16 rounding', 'Bit-exact'],
              ['Random activations, 1 to 8 rows, against float64 products of the reference decode', 'Relative error ≤ 2.3×10<super>−7</super>'],
              ['NumPy ports of the PyTorch GGUF decoders vs the official gguf 0.19 dequantizer', 'Bit-exact'],
          ], widths=(4.9 * inch, 1.8 * inch)),
          H2('Next-layer expert prediction'),
          P('Routed experts for layer N+1 used to be requested only when layer N+1 routed, so the drive sat idle '
            'while the GPU ran layer N’s experts and layer N+1’s attention. In v0.6, during a decode step, '
            'layer N’s MoE input is also scored by layer N+1’s router, and reads for its six best experts are '
            'queued behind layer N’s own reads. When layer N+1 routes for real, correct guesses are already read '
            'or in flight; guesses not yet started are cancelled. The real router always decides which experts run, '
            'so a wrong guess costs drive time and never changes the output. The share of routed experts that were '
            'guessed (recall) is recorded in the run evidence.'),
          H2('Status'),
          P('Both features are on by default, in frustum_chat and in ActivatePrime. They were verified without a GPU '
            'as above; speed on the RTX 5060 Ti and prediction recall have not been measured yet. run_fused_check.ps1 '
            'times each kernel against the reference on random weights, and run_speed_check.ps1 runs each model '
            'before and after in fresh processes, reporting seconds per token, whether the token ids match and '
            'DeepSeek’s recall. Results will be added to this section when recorded.'),
          PageBreak()]

    s += [H1('16. Interpretation and limitations'),
          H2('Bandwidth first'),
          P('An all-miss pass through six routed FP4 experts in each of 40 layers requests about 4.5 GB. At 50 MB/s '
            'that is roughly 90 s of transfer; at 500 MB/s it is roughly 9 s. Expert reuse, the GPU and host tiers, '
            'and resident dense weights bring actual traffic well below the all-miss figure, which is why the live '
            'DeepSeek decode lands near 8 s per token. Further gains are most likely from reducing bytes per token '
            '(better reuse, prediction, layout) rather than from faster arithmetic.'),
          H2('Threats to measurement'),
          P('Logical read counters measure bytes requested from files, not physical device I/O. OS caching may satisfy '
            'reads. The live v0.5.1 numbers are single runs from the app, without per-token timing breakdowns or '
            'controlled cache state. First-use GPU initialization and validation work contaminate elapsed time in the '
            'older runs. The USB comparison changes the link only; it was not repeated in reverse order.'),
          H2('Still unverified'),
          P('Full-model numerical parity against the reference implementation, the v0.5 reader and placement against v0.4 '
            'on the same day (run_verify.ps1), long-context behaviour (run_long_context.ps1), other GPUs and drives, '
            'the v0.6 speed-ups and prediction recall on the target GPU, and multi-turn quality at length.'),
          H2('Position relative to earlier work'),
          P('On-demand flash-backed model execution is not a new general concept; LLM in a Flash studies memory-limited '
            'inference that reduces transfer volume through reuse and storage-aware access. [W3] Frustum_loader '
            'contributes a specific, working implementation and reproducibility record for this local mixed-format '
            'checkpoint on a consumer GPU. No novelty, priority or comparative speedup claim against other systems is '
            'made by these tests.'),
          PageBreak()]

    # ---------------------------------------------------------------- 16
    s += [H1('17. Reproduction and integration'),
          H2('Environment'),
          P('Launchers look for Python in this order: FRUSTUM_PYTHON, .venv\\Scripts\\python.exe in the project folder, '
            'then the original Codex environment. A fresh machine:'),
          code('py -3.13 -m venv .venv\n'
               '.venv\\Scripts\\pip install torch --index-url https://download.pytorch.org/whl/cu128\n'
               '.venv\\Scripts\\pip install -r requirements.txt'),
          H2('Chat and verification'),
          code('cd G:\\Models\\DeepSeek-V4.1-Flash\\Frustum_loader\n'
               '.\\run_verify.ps1                      # tests, drive speed, v0.4 parity, timed chat\n'
               '.\\run_chat.ps1                        # interactive chat\n'
               '.\\run_chat.ps1 --prompt "Say hello." --max-new-tokens 6\n'
               '.\\run_chat.ps1 --legacy               # v0.4 placement and mmap reader'),
          H2('v0.6 speed checks'),
          code('.\\run_fused_check.ps1                 # each kernel vs the reference, timed, no model weights\n'
               '.\\run_speed_check.ps1                 # Phi-4 and GLM before/after, s/token and token match\n'
               '.\\run_speed_check.ps1 --deepseek      # adds DeepSeek with prediction recall (long)'),
          H2('Component tests'),
          code('.\\run_expert_test.ps1\n.\\run_moe_test.ps1\n.\\run_sparse_rows_test.ps1\n'
               '.\\run_block_primitives_test.ps1\n.\\run_layers0_3_test.ps1\n.\\run_repro_test.ps1\n'
               '.\\run_glm_placement_test.ps1 --evidence placement-a.json\n'
               '.\\run_glm_placement_test.ps1 --order "split,baseline" --evidence placement-b.json'),
          H2('Configuration'),
          table([
              ['Variable', 'Purpose'],
              ['FRUSTUM_PYTHON', 'CUDA-enabled python.exe to use'],
              ['FRUSTUM_MODEL / FRUSTUM_MODELS', 'DeepSeek checkpoint folder / models root for ActivatePrime'],
              ['FRUSTUM_LOADER', 'Frustum_loader folder used by ActivatePrime\u2019s worker'],
              ['FRUSTUM_READER', 'Set to mmap for the legacy reader'],
              ['FRUSTUM_DEEPSEEK_HOST_GIB', 'Host-RAM expert tier size'],
              ['FRUSTUM_GLM_PLACEMENT', 'baseline, split (default) or tiered'],
              ['FRUSTUM_FUSED', '0 turns off the v0.6 fused decode kernels'],
              ['FRUSTUM_PREDICT_NEXT', 'Next-layer experts to read early per decode layer (default 6, 0 off)'],
          ]),
          H2('ActivatePrime'),
          P('ActivatePrime is the Electron chat front end. It keeps one Python worker warm, streams tokens, offers '
            'Phi-4, GLM-4.7-Flash and DeepSeek V4.1 Flash as backends and frees the model when stopped. Weights and the '
            'CUDA environment are not bundled with either project.'),
          PageBreak()]

    # ---------------------------------------------------------------- 17
    s += [H1('18. Contributors and references'),
          table([
              ['Contributor', 'Role'],
              ['Aaron Grace', 'Project concept, architecture, original engine, hardware testing and direction'],
              ['OpenAI (ChatGPT, Codex)', 'Implementation and analysis for v0.1 to v0.4; earlier whitepaper editions'],
              ['Claude (Anthropic)', 'v0.5 and v0.6 engineering: RangeReader, tiered expert cache, eviction policy, placement, prefix reuse, streaming, fused decode kernels, next-layer prediction, this edition'],
              [f'Dino Buddy {DINO}', 'Contributor and test prompt'],
              ['DinoClaw', 'Aaron Grace\u2019s original agent runtime; assistance'],
          ], widths=(1.9 * inch, 4.8 * inch)),
          H2('Local evidence'),
          P('[L1] Local config.json, model.safetensors.index.json and shard headers. [L2] Local README.md model card. '
            '[L3] Local inference/model.py and inference/kernel.py. [L4] moe_correctness_results.json. '
            '[L5] expert_correctness_results.json. [L6] sparse_rows_results.json. [L7] block_primitives_results.json.'),
          H2('External references'),
          P('[W1] NVIDIA NeMo AutoModel: MXFP4 packing and decoding documentation. '
            'docs.nvidia.com/nemo/automodel/nightly/nemo-automodel/nemo_automodel/components/quantization/mxfp4'),
          P('[W2] PyTorch: CUDA semantics and pinned-memory / non-blocking transfer documentation. '
            'docs.pytorch.org/docs/stable/notes/cuda.html'),
          P('[W3] Alizadeh et al., LLM in a flash: Efficient Large Language Model Inference with Limited Memory, '
            'ACL 2024; arXiv:2312.11514.'),
          H2('Version history'),
          P('0.1 four MoE sublayers. 0.2 Layer 0 block. 0.3 repeatability audit and Layers 0\u20133. 0.3.8 first '
            '40-layer reply, resident head, expert cache benchmark, Phi-4 and GLM backends. 0.4.0 GLM placement and '
            'warm chat. 0.5.1 new reader, tiered cache, placement plan, live ActivatePrime benchmarks on USB-C, and the '
            'rename from Frusium_loader to Frustum_loader. 0.6.0 fused decode kernels for every backend and next-layer '
            'expert prediction for DeepSeek.'),
          PageBreak()]

    # ---------------------------------------------------------------- appendix
    s += [H1('Appendix A. Run evidence'),
          P('The original files below are embedded in this PDF. Open the Attachments pane in a PDF reader to save copies. '
            'Some older records contain machine-specific paths and the former project name.'),
          table([['Attachment', 'Contents']] + [[n, d] for n, d in EVIDENCE if (ROOT / n).exists()]),
          ]
    return s


EVIDENCE = [
    ('chat_smoke_results.json', 'Original 40-layer generation'),
    ('chat_resident_head_results.json', '40-layer rerun with per-token timings'),
    ('layers0_3_results.json', 'Sequential correctness and memory checks'),
    ('reproducibility_results.json', 'Layer 0 reproducibility report'),
    ('repro_baseline.json', 'Saved Layer 0 baseline'),
    ('decode_profile_results.json', '40-layer profile with 256 MiB expert cache'),
    ('decode_profile_6144_results.json', '40-layer profile with 6 GiB expert cache'),
    ('dino_buddies_chat_results.json', 'Dino Buddies prompt, reply and measurements'),
    ('phi4_chat_results.json', 'Phi-4 prompt, reply, token IDs and timings'),
    ('glm_chat_results.json', 'GLM-4.7-Flash prompt, reply and measurements'),
    ('glm_placement_results.json', 'GLM placement comparison'),
    ('glm_placement_reverse_results.json', 'GLM placement comparison, reverse order'),
    ('moe_correctness_results.json', 'MoE sublayer correctness'),
    ('expert_correctness_results.json', 'Single-expert correctness'),
    ('sparse_rows_results.json', 'Sparse row lookups'),
    ('block_primitives_results.json', 'Residual primitives'),
    ('model_inspection.json', 'Checkpoint catalog summary'),
]


def build():
    tmp = OUT.with_suffix('.tmp.pdf')
    doc = SimpleDocTemplate(str(tmp), pagesize=letter, leftMargin=0.9 * inch, rightMargin=0.9 * inch,
                            topMargin=0.95 * inch, bottomMargin=0.85 * inch,
                            title='Frustum_loader Whitepaper v0.6.0', author='Aaron Grace',
                            subject='SSD-streamed mixture-of-experts inference on a 16 GB GPU')
    doc.build(story(), onFirstPage=on_page, onLaterPages=on_page)
    from pypdf import PdfReader, PdfWriter
    w = PdfWriter(clone_from=PdfReader(str(tmp)))
    for name, _ in EVIDENCE:
        p = ROOT / name
        if p.exists():
            w.add_attachment(name, p.read_bytes())
    w.add_metadata({'/Title': 'Frustum_loader Whitepaper v0.6.0', '/Author': 'Aaron Grace'})
    with open(OUT, 'wb') as f:
        w.write(f)
    tmp.unlink()
    print(OUT)


if __name__ == '__main__':
    build()
