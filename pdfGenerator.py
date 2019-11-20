from fpdf import FPDF


data = {'Monogram': {0: 'site', 1: 'referencement', 2: 'google', 3: 'web', 4: 'seo', 5: 'recherche', 6: 'page', 7: 'mots', 8: 'pages', 9: 'contenu', 10: 'optimiser', 11: 'cles', 12: 'internet', 13: 'marketing', 14: 'naturel', 15: 'liens', 16: 'sites', 17: 'moteurs', 18: 'resultats', 19: 'wordpress', 20: 'images', 21: 'temps', 22: 'agence', 23: 'blog', 24: 'strategie', 25: 'article', 26: 'mobile', 27: 'pouvez', 28: 'exemple', 29: 'chargement', 30: 'trafic', 31: 'qualite', 32: 'commerce', 33: 'titre', 34: 'articles', 35: 'egalement', 36: 'formation', 37: 'cle', 38: 'creer', 39: 'ameliorer'}, 'Freq_Mono': {0: 2181, 1: 1188, 2: 1160, 3: 1031, 4: 919, 5: 787, 6: 764, 7: 672, 8: 597, 9: 592, 10: 551, 11: 548, 12: 397, 13: 370, 14: 369, 15: 321, 16: 317, 17: 313, 18: 307, 19: 305, 20: 291, 21: 285, 22: 257, 23: 248, 24: 245, 25: 228, 26: 226, 27: 225, 28: 222, 29: 212, 30: 212, 31: 209, 32: 201, 33: 196, 34: 194, 35: 191, 36: 189, 37: 188, 38: 187, 39: 184}, 'Bigram': {0: 'mots cles', 1: 'site web', 2: 'referencement naturel', 3: 'site internet', 4: 'moteurs recherche', 5: 'recherche vocale', 6: 'optimiser site', 7: 'mot cle', 8: 'reseaux sociaux', 9: 'temps chargement', 10: 'optimiser referencement', 11: 'moteur recherche', 12: 'resultats recherche', 13: 'referencement local', 14: 'referencement site', 15: 'longue traine', 16: 'referencement seo', 17: 'site commerce', 18: 'mis jour', 19: 'vitesse chargement'}, 'Freq_Bi': {0: 506, 1: 470, 2: 343, 3: 264, 4: 233, 5: 163, 6: 158, 7: 147, 8: 137, 9: 104, 10: 103, 11: 96, 12: 90, 13: 85, 14: 83, 15: 82, 16: 80, 17: 77, 18: 73, 19: 70}, 'Trigram': {0: 'referencement naturel seo', 1: 'optimiser site web', 2: 'site recherche vocale', 3: 'optimiser site recherche', 4: 'google search console', 5: 'referencement naturel site', 6: 'temps chargement site', 7: 'referencement naturel referencement', 8: 'mot cle principal', 9: 'cles longue traine'}, 'Freq_Tri': {0: 58, 1: 52, 2: 29, 3: 28, 4: 26, 5: 25, 6: 25, 7: 24, 8: 23, 9: 21}}

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
    
def add_monogram(pdf,data,ln):
    for i in range(len(data["Monogram"])):
        pdf.cell(50,5, txt=data["Monogram"][i], ln= 1, align="L")
    return pdf

def generate_pdf():
    pdf = PDFGenerator()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Welcome to Python!", ln=1, align="C")
    pdf.set_font('Times', '', 12)
    return pdf
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
print(len(data["Monogram"]))
add_monogram(s,data,2).output("test.pdf")