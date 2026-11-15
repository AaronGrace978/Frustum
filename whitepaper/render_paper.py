import json,html
from base64 import b64encode
import sys
from pathlib import Path
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,PageBreak,Table,TableStyle,Preformatted
from reportlab.platypus import Image
from io import BytesIO
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_LEFT
root=Path(__file__).parent
pdfmetrics.registerFont(TTFont('Body',r'C:/Windows/Fonts/arial.ttf'))
pdfmetrics.registerFont(TTFont('Bold',r'C:/Windows/Fonts/arialbd.ttf'))
pdfmetrics.registerFontFamily('Body',normal='Body',bold='Bold')
styles={
'p':ParagraphStyle('Body',fontName='Body',fontSize=10.2,leading=14.3,spaceAfter=9,textColor=colors.HexColor('#263446')),
'h':ParagraphStyle('Heading',fontName='Bold',fontSize=12,leading=16,spaceBefore=9,spaceAfter=6,textColor=colors.HexColor('#087D89')),
'title':ParagraphStyle('Title',fontName='Bold',fontSize=24,leading=29,spaceAfter=20,textColor=colors.HexColor('#13283F')),
'sub':ParagraphStyle('Sub',fontName='Body',fontSize=16,leading=22,spaceAfter=20,textColor=colors.HexColor('#087D89')),
'cell':ParagraphStyle('Cell',fontName='Body',fontSize=9,leading=12),
'code':ParagraphStyle('Code',fontName='Courier',fontSize=9,leading=14,spaceAfter=12)}
current_edition = '--current' in sys.argv
glm_edition = current_edition or '--v038' in sys.argv
phi_edition = glm_edition or '--v037' in sys.argv
dino_edition = phi_edition or '--v036' in sys.argv
cache_edition = dino_edition or '--v035' in sys.argv
evidence_edition = cache_edition or '--v034' in sys.argv
benchmark_edition = evidence_edition or '--v033' in sys.argv
contributor_edition = benchmark_edition or '--v032' in sys.argv
illustrated = contributor_edition or '--v031' in sys.argv
new_edition = illustrated or '--v03' in sys.argv
version = '0.4.0' if current_edition else '0.3.8' if glm_edition else '0.3.7' if phi_edition else '0.3.6' if dino_edition else '0.3.5' if cache_edition else '0.3.4' if evidence_edition else '0.3.3' if benchmark_edition else '0.3.2' if contributor_edition else '0.3.1' if illustrated else '0.3'
source = 'paper_v0.2_draft.json' if new_edition else 'paper.json'
pages=json.loads((root/source).read_text(encoding='utf8'))
if new_edition:
    pages.extend(json.loads((root/'milestone_v0.3.json').read_text(encoding='utf8')))
    for page in pages:
        page['body'] = [(kind, text.replace(
            'Cross-session reproducibility is the next acceptance gate.',
            'The later reproducibility audit and sequential results are recorded in section 7.'
        ).replace(
            'This revision adds the layer-0 milestone and reproducibility audit; it does not validate later layers.',
            'Version 0.2 recorded the Layer 0 milestone. Version 0.3 adds the saved-baseline repeatability result and sequential validation through Layer 3.'
        ).replace('Version 0.2 | November 13, 2026', f'Version {version} | November 14, 2026') if isinstance(text, str) else text) for kind, text in page['body']]
if contributor_edition:
    from PIL import Image as PillowImage, ImageDraw, ImageFont

    icon = PillowImage.new('RGBA', (96, 96))
    font = ImageFont.truetype('C:/Windows/Fonts/seguiemj.ttf', 72)
    ImageDraw.Draw(icon).text((48, 48), '🦖', font=font, anchor='mm', embedded_color=True)
    icon = icon.crop(icon.getbbox())
    credits_icon = BytesIO()
    icon.save(credits_icon, format='PNG')
    icon_uri = 'data:image/png;base64,' + b64encode(credits_icon.getvalue()).decode('ascii')
    pages[0]['body'].insert(2, ('contributor', 'Contributor: Dino Buddy. The project also received assistance from DinoClaw, Aaron Grace’s original agent runtime.'))
if illustrated:
    pages.append({
        'title': '8. First local reply',
        'body': [
            ('figure', 'first_local_reply.png'),
            ('p', 'Figure 1. Visualization of the first local generation run, made from chat_smoke_results.json. The checkpoint answered “Say hello.” with “Hello!” followed by a waving-hand emoji and “How can” before the six-token limit.'),
        ],
    })
if benchmark_edition:
    original = json.loads((root.parent / 'chat_smoke_results.json').read_text())
    resident = json.loads((root.parent / 'chat_resident_head_results.json').read_text())
    timings = resident['token_timings']
    matches = original['token_ids'] == resident['token_ids']
    decode = timings[1:]
    pages.append({
        'title': '9. GPU-resident output head',
        'body': [
            ('p', 'This experiment repeated the “Say hello.” prompt with a six-token limit. The output head was loaded once into GPU memory in BF16. Scoring retained FP32 arithmetic and the original 4,096-row chunks. The streamed output head remains available.'),
            ('table', [
                ['Measurement', 'Original streamed / GPU-resident'],
                ['Generation wall time', f"{original['seconds']:.2f} / {resident['seconds']:.2f} s"],
                ['Logical output-head reads', f"{original['output_head_logical_bytes']/1e9:.3f} / {resident['output_head_logical_bytes']/1e9:.3f} GB"],
                ['Peak PyTorch allocated memory', f"{original['peak_allocated_GiB']:.3f} / {resident['peak_allocated_GiB']:.3f} GiB"],
                ['Peak PyTorch reserved memory', f"{original['peak_reserved_GiB']:.3f} / {resident['peak_reserved_GiB']:.3f} GiB"],
            ]),
            ('p', f"Generated token IDs {'matched exactly' if matches else 'differed between runs'}. The resident run reached its first token in {timings[0]['total_seconds']:.2f} s, including {timings[0]['output_head_seconds']:.2f} s for normalization, loading and scoring."),
            ('p', f"Subsequent tokens averaged {sum(t['total_seconds'] for t in decode)/len(decode):.2f} s each. Output-head work averaged {sum(t['output_head_seconds'] for t in decode)/len(decode):.3f} s and requested {sum(t['head_logical_bytes'] for t in decode):,} additional logical head bytes."),
            ('p', 'Each mode has one recorded run. The original run lacks per-token timings, and the operating-system cache state was uncontrolled. Wall times describe those runs. Logical reads count requested payload bytes and do not measure physical SSD traffic.'),
            ('p', 'Evidence: chat_smoke_results.json and chat_resident_head_results.json.'),
        ],
    })
if cache_edition:
    small, large = [
        json.loads((root.parent / name).read_text(encoding='utf-8'))
        for name in ('decode_profile_results.json', 'decode_profile_6144_results.json')
    ]
    def step(run, index):
        return run['token_timings'][index]

    def expert_gb(run, index):
        return sum(l['logical_bytes']['experts'] for l in step(run, index)['layers']) / 1e9

    pairs = [small, large]
    matches = small['token_ids'] == large['token_ids'] == original['token_ids'][:3]
    pages.append({
        'title': '10. Expert cache benchmark',
        'body': [
            ('p', 'Both runs used the prompt “Say hello.”, a three-token limit, a GPU-resident output head, and synchronized per-layer decode profiling across all 40 layers.'),
            ('table', [
                ['Measurement', '256 MiB / 6 GiB expert cache'],
                ['Time to first token', ' / '.join(f"{step(r, 0)['total_seconds']:.2f} s" for r in pairs)],
                ['First cached decode', ' / '.join(f"{step(r, 1)['total_seconds']:.2f} s" for r in pairs)],
                ['Second cached decode', ' / '.join(f"{step(r, 2)['total_seconds']:.2f} s" for r in pairs)],
                ['First decode expert reads', ' / '.join(f'{expert_gb(r, 1):.3f} GB' for r in pairs)],
                ['Second decode expert reads', ' / '.join(f'{expert_gb(r, 2):.3f} GB' for r in pairs)],
                ['Peak PyTorch reserved', ' / '.join(f"{r['peak_reserved_GiB']:.3f} GiB" for r in pairs)],
            ]),
            ('p', f"Token IDs {'matched the original generation prefix' if matches else 'differed between runs'}. The second cached decode requested {expert_gb(small, 2) - expert_gb(large, 2):.3f} GB fewer logical expert bytes with the larger cache."),
            ('p', 'Each cache size has one measured run. OS cache state was uncontrolled. Logical reads measure checkpoint payload requests. The experiment covers one prompt and two cached decode steps.'),
            ('p', 'Both per-layer profiles are embedded in Appendix A.'),
        ],
    })
if dino_edition:
    dino_record_name = 'dino_buddies_chat_results.json'
    dino = json.loads((root.parent / dino_record_name).read_text(encoding='utf-8'))
    if dino['prompt'] != 'Do you like Dino Buddies? 🦖' or not dino['reply'].strip():
        raise ValueError('Unexpected Dino Buddies run evidence')
    pages.append({
        'title': '11. Dino Buddies chat demo',
        'body': [
            ('h', 'Prompt'),
            ('transcript', dino['prompt']),
            ('h', 'Model reply'),
            ('transcript', dino['reply']),
            ('h', 'Run settings and measurements'),
            ('p', f"Local greedy generation across {dino['layers']} layers with a resident output head and a {dino['expert_cache_mib']} MiB expert cache. The run generated {len(dino['token_ids'])} text tokens out of a 24-token limit in {dino['seconds']:.2f} s. Peak PyTorch-reserved memory was {dino['peak_reserved_GiB']:.3f} GiB."),
            ('p', 'The exact Unicode prompt and reply, token IDs, and timings are in the embedded run record.'),
        ],
    })
if phi_edition:
    from statistics import mean

    phi_record_name = 'phi4_chat_results.json'
    phi = json.loads((root.parent / phi_record_name).read_text(encoding='utf-8'))
    pages.append({
        'title': '12. Phi-4 on Frustum_loader',
        'body': [
            ('p', 'Frustum_loader loaded Microsoft Phi-4 14B Q4_K_S GGUF into GPU memory in packed form and ran all 40 layers in PyTorch on an RTX 5060 Ti (16 GB).'),
            ('h', 'Prompt'),
            ('transcript', phi['prompt']),
            ('h', 'Model reply'),
            ('transcript', phi['reply']),
            ('p', 'The smoke test stopped at the 24-token limit. The toy-line description is model output.'),
            ('table', [
                ['Measurement', 'Observed result'],
                ['Packed weight load', f"{phi['load_seconds']:.2f} s"],
                ['First token after loading', f"{phi['first_token_seconds']:.2f} s"],
                ['Mean cached decode', f"{mean(phi['decode_seconds']):.2f} s per token"],
                ['Peak PyTorch-reserved GPU memory', f"{phi['peak_reserved_GiB']:.2f} GiB"],
            ]),
            ('p', 'The original JSON is embedded in Appendix A. Sampled dequantization checks and finite-logit checks passed. Full-model numerical parity remains unverified.'),
        ],
    })
if glm_edition:
    glm_record_name = 'glm_chat_results.json'
    glm = json.loads((root.parent / glm_record_name).read_text(encoding='utf-8'))
    pages.append({
        'title': '13. GLM-4.7-Flash on Frustum_loader',
        'body': [
            ('p', 'The GLM backend executed all 47 layers using Frustum_loader’s bounded reader and PyTorch CUDA. It loaded routed quantized expert slices on demand.'),
            ('h', 'Prompt'),
            ('transcript', glm['prompt']),
            ('h', 'Model reply'),
            ('transcript', glm['reply']),
            ('p', f"The run generated {len(glm['token_ids'])} tokens and {'reached an end token' if glm['stopped_at_eos'] else 'reached the token limit'}."),
            ('table', [
                ['Measurement', 'Observed result'],
                ['First token, including on-demand reads', f"{glm['first_token_seconds']:.2f} s"],
                ['Mean cached decode', f"{mean(glm['decode_seconds']):.2f} s per step"],
                ['Peak PyTorch-reserved GPU memory', f"{glm['peak_reserved_GiB']:.2f} GiB"],
                ['Logical weight payload reads', f"{glm['logical_weight_read_bytes']/1e9:.2f} GB"],
            ]),
            ('p', 'This single-prompt smoke test verifies generation with this backend. Full numerical parity and multi-turn performance remain unverified. The exact run record is embedded in Appendix A.'),
        ],
    })
if evidence_edition:
    from hashlib import sha256
    from pypdf import PdfReader, PdfWriter

    repeat = 'repro_audit/repeat_20261114_024827'
    evidence_files = [
        ('chat_smoke_results.json', 'Original 40-layer generation'),
        ('chat_resident_head_results.json', '40-layer rerun with per-token timings'),
        ('layers0_3_results.json', 'Sequential correctness and memory checks'),
        ('reproducibility_results.json', 'Layer 0 reproducibility report'),
        ('repro_baseline.json', 'Saved Layer 0 baseline'),
        (f'{repeat}/run_1.log', 'Layer 0 repeat 1 terminal output'),
        (f'{repeat}/run_2.log', 'Layer 0 repeat 2 terminal output'),
        (f'{repeat}/run_1.json', 'Layer 0 repeat 1 structured result'),
        (f'{repeat}/run_2.json', 'Layer 0 repeat 2 structured result'),
    ]
    if cache_edition:
        evidence_files.extend([
            ('decode_profile_results.json', '40-layer profile with 256 MiB expert cache'),
            ('decode_profile_6144_results.json', '40-layer profile with 6 GiB expert cache'),
        ])
    if dino_edition:
        evidence_files.append((dino_record_name, 'Dino Buddies prompt, exact reply, and run measurements'))
    if phi_edition:
        evidence_files.append((phi_record_name, 'Phi-4 prompt, exact reply, token IDs, and timings'))
    if glm_edition:
        evidence_files.append((glm_record_name, 'GLM-4.7-Flash prompt, reply, and measurements'))
    if current_edition:
        evidence_files.extend([
            ('glm_placement_results.json', 'Unified, split, and tiered placement results'),
            ('glm_placement_reverse_results.json', 'Reverse-order placement repeat'),
        ])
    attachments = {Path(path).name: (root.parent / path).read_bytes() for path, _ in evidence_files}
    pages.append({
        'title': 'Appendix A. Run evidence',
        'body': [
            ('p', 'The original files below are embedded in this PDF. Open the Attachments pane in a PDF reader to save copies.'),
            ('table', [['Attachment', 'Contents']] + [[Path(path).name, description] for path, description in evidence_files]),
            ('h', 'Generation record excerpt'),
            ('code', f"Prompt: {original['prompt']}\nToken IDs: {original['token_ids']}\nTokens match across the two runs: {original['token_ids'] == resident['token_ids']}"),
            ('p', 'The .log attachments contain the Layer 0 terminal transcripts. The two 40-layer generation runs are preserved as structured JSON records.'),
        ],
    })
if current_edition:
    from current_content import current_pages
    pages = current_pages(root)
story=[]
for idx,page in enumerate(pages):
    if idx:story.append(PageBreak())
    story.append(Paragraph(html.escape(page['title']),styles['title']))
    if 'subtitle'in page:story.append(Paragraph(html.escape(page['subtitle']),styles['sub']))
    for kind,text in page['body']:
        if kind=='table':
            rows=[[Paragraph(html.escape(c),styles['cell']) for c in row] for row in text]
            table=Table(rows,colWidths=[190,314],hAlign='LEFT')
            table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#D6EEF0')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),8),('RIGHTPADDING',(0,0),(-1,-1),8),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),('LINEBELOW',(0,0),(-1,-1),.4,colors.HexColor('#D8E0E7'))]))
            story.extend([table,Spacer(1,12)])
        elif kind=='code':story.append(Preformatted(text,styles['code']))
        elif kind=='transcript':
            # Paint emoji inline so ReportLab can preserve them alongside the text.
            import re
            pieces = re.split(r'([\U0001F000-\U0001FAFF](?:\ufe0f)?)', text)
            for index in range(1, len(pieces), 2):
                emoji = PillowImage.new('RGBA', (96, 96))
                ImageDraw.Draw(emoji).text((48, 48), pieces[index], font=font, anchor='mm', embedded_color=True)
                buffer = BytesIO()
                emoji.crop(emoji.getbbox()).save(buffer, format='PNG')
                uri = 'data:image/png;base64,' + b64encode(buffer.getvalue()).decode('ascii')
                pieces[index] = f'<img src="{uri}" width="13" height="13" valign="middle"/>'
            transcript = ''.join(html.escape(piece) if i % 2 == 0 else piece for i, piece in enumerate(pieces))
            story.append(Paragraph(transcript.replace('\n', '<br/>'), styles['p']))
        elif kind=='contributor':
            label, details = text.split('. ', 1)
            story.append(Paragraph(
                f'{html.escape(label)} <img src="{icon_uri}" width="12" height="12" valign="middle"/>. {html.escape(details)}',
                styles['p'],
                caseSensitive=1,
            ))
        elif kind=='figure':
            figure = Image(str(root.parent / text), width=492, height=492 * 530 / 1200)
            story.extend([figure, Spacer(1, 12)])
        else:story.append(Paragraph(html.escape(text),styles[kind]))
def frame(canvas,doc):
    canvas.setStrokeColor(colors.HexColor('#087D89'));canvas.setLineWidth(2);canvas.line(54,753,558,753)
    canvas.setFont('Body',8);canvas.setFillColor(colors.HexColor('#64748B'));canvas.drawString(54,766,'FRUSTUM_LOADER / EXPERIMENTAL WHITEPAPER')
    footer = f'Version {version} | November 14, 2026 | Sequential validation' if new_edition else 'Version 0.2 | November 13, 2026 | Component-level validation'
    canvas.drawString(54,30,footer)
    canvas.drawRightString(558,30,str(doc.page))
output = root / f'Frustum_loader_whitepaper_v{version if new_edition else "0.2"}.pdf'
if current_edition:
    output = root / 'Frustum_loader_Whitepaper.pdf'
destination = BytesIO() if evidence_edition else str(output)
SimpleDocTemplate(destination,pagesize=(612,792),leftMargin=54,rightMargin=54,topMargin=61,bottomMargin=52,title='Frustum_loader: Bounded Working-Set Execution',author='Aaron Grace; implementation assistance by ChatGPT').build(story,onFirstPage=frame,onLaterPages=frame)
if evidence_edition:
    writer = PdfWriter(clone_from=PdfReader(destination))
    for name, data in attachments.items():
        writer.add_attachment(name, data)
    writer.write(output)
    embedded = PdfReader(output).attachments
    if {name: sha256(data[0]).digest() for name, data in embedded.items()} != {
        name: sha256(data).digest() for name, data in attachments.items()
    }:
        raise RuntimeError('PDF attachments did not match the original evidence files')
    print(f'Verified {len(attachments)} embedded evidence files.')
if not new_edition:
    import fitz
    doc=fitz.open(output)
    print('Pages:',len(doc))
    for i,page in enumerate(doc):
        page.get_pixmap(matrix=fitz.Matrix(1,1)).save(root/f'page_{i+1}.png')
        print(i+1,len(page.get_text()),page.get_text().splitlines()[:3])
