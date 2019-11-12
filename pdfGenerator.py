from fpdf import FPDF

class PDFGenerator(FPDF):
    
    def header(self):
        # Set up a logo
        self.image('images/devlup.png', 10, 8, 40, 13)
        self.set_line_width(0.5)
        self.set_draw_color(20, 20, 20)
        self.line(0, 25, 210, 25)
 
        # Line break
        self.ln(20)
 
    def footer(self):
        self.set_y(-10)
 
        self.set_font('Arial', 'I', 8)
 
        # Add a page number
        page = 'Page ' + str(self.page_no()) + '/{nb}'
        self.cell(0, 10, page, 0, 0, 'C')


def generate_pdf():
    pdf = PDFGenerator()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Welcome to Python!", ln=1, align="C")
    pdf.set_font('Times', '', 12)
    pdf.output("Test.pdf")
def change_fonts():
    pdf = PDFGenerator()
    pdf.add_page()
    font_size = 8
    for font in pdf.core_fonts:
        if any([letter for letter in font if letter.isupper()]):
            # skip this font
            continue
        pdf.set_font(font, size=font_size)
        txt = "Font name: {} - {} pts".format(font, font_size)
        pdf.cell(0, 10, txt=txt, ln=1, align="C")
        font_size += 2
    
        pdf.output("change_fonts.pdf")
def draw_lines():
    pdf = PDFGenerator()
    pdf.add_page()
    
    pdf.output('draw_lines.pdf')

def add_image(image_path):
    pdf = PDFGenerator()
    pdf.add_page()
    pdf.image(image_path, x=10, y=8, w=100)
    pdf.set_font("Arial", size=12)
    pdf.ln(85)  # move 85 down
    pdf.cell(200, 10, txt="{}".format(image_path), ln=1)
    pdf.output("add_image.pdf")


s = generate_pdf()
ss = add_image("images/devlup.png")