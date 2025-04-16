from pdf2image import convert_from_path

pages = convert_from_path("/opt/data/private/yj_data/IMAGGarment_weight/code/assets/architecture.pdf", dpi=300)
for i, page in enumerate(pages):
    page.save(f"page_{i+1}.png", "PNG")
