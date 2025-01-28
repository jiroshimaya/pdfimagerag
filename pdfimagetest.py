import base64

import dotenv
import fitz  # pymupdf
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

dotenv.load_dotenv()


def convert_pdf_to_base64_images(pdf_path: str) -> list[str]:
    doc = fitz.open(pdf_path)
    base64_images_by_page = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap()
        base64_image = base64.b64encode(pix.tobytes()).decode("utf-8")
        base64_images_by_page.append(base64_image)
    return base64_images_by_page


def ask_gpt(text: str, base64_image: str) -> str:
    model = ChatOpenAI(model="gpt-4o-mini")
    message = HumanMessage(
        content=[
            {"type": "text", "text": text},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
            },
        ],
    )
    response = model.invoke([message])
    return response.content


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PDFのパスを取得")
    parser.add_argument("pdf_path", help="PDFファイルのパス")
    parser.add_argument("--page_num", type=int, default=0, help="PDFのページ番号")
    parser.add_argument(
        "--question",
        type=str,
        default="この画像は全体的に何色ですか",
        help="GPTに尋ねる質問",
    )
    args = parser.parse_args()
    pdf_path = args.pdf_path

    base64_images_by_page = convert_pdf_to_base64_images(pdf_path)
    image_data = base64_images_by_page[args.page_num]
    response = ask_gpt(args.question, image_data)
    print(response)
