print('Iniciando...')
import cv2
from ultralytics import YOLO
import os
import sys
import webbrowser
import graph_creator

from PIL import Image
import google.generativeai as genai
import os

# Tentar carregar variáveis de ambiente do arquivo .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv não está instalado, usar apenas variáveis de ambiente do sistema

# Configurar API key do Google Generative AI
api_key = os.getenv('GOOGLE_API_KEY')
if not api_key:
    raise ValueError(
        "GOOGLE_API_KEY não encontrada!\n"
        "Configure a variável de ambiente ou crie um arquivo .env\n"
        "Veja o README.md para instruções detalhadas."
    )

genai.configure(api_key=api_key)
MODEL_ID = "gemini-2.0-flash"
model = genai.GenerativeModel(MODEL_ID)



output_folder = './results/'
# Verifica se a pasta de saída existe, caso contrário, cria
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def clear():
    if os.name == 'nt':  # 'nt' é para Windows
        os.system('cls')
    else:  # 'posix' é para sistemas Unix/Linux (incluindo macOS)
        os.system('clear')

def start():
    clear()
    print(""" Bem vindo à
                 𝔼𝕕𝕦𝕍𝕚𝕤𝕚𝕠𝕟   𝕀𝔸 \n""")

def import_img():
    image_path = input('Diga o caminho da imagem: ')
    if (os.path.exists(image_path)):
        return image_path
    else:
        start()
        print('Caminho não existe! \n')
        import_img()

def cut_image(image_path):
    print('Buscando gráficos na imagem')
    model = YOLO("./MLs/ML1.pt")
    image = cv2.imread(image_path)
    results = model(image_path)
    result = results[0]
    graph_counter = 0
    for i, box in enumerate(result.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        if label == 'grafico':
            conf = float(box.conf[0])
            # Recorte do objeto
            cropped = image[y1:y2, x1:x2]
            # Criar nome de arquivo de saída
            output_path = os.path.join(output_folder, f"Grafico_{graph_counter+1}.jpg")
            # Salvar recorte
            cv2.imwrite(output_path, cropped)
            graph_counter += 1
    clear()
    return graph_counter

def save(graph_images):
    print(f"Foram salvos {graph_images} gráficos")
    
def listar_graficos():
    listar_graficos = input('Deseja listar os gráficos? (y ou n): ')
    clear()
    print('Gráficos: \n')
    if listar_graficos == 'y':
        graficos = os.listdir(output_folder)
        graficos.sort()
        count = 0
        for i in graficos:
            count = count+1
            print(f'{count}. {i}')
        return count


def selecionar_grafico(quantidade):
    grafico = int(input('Digite o gráfico para abrir: '))
    if grafico >0 and grafico <= quantidade:
        clear()
        print(f'Gráfico {grafico} selecionado')
    else:
        print('Gráfico não encontrado, tente novamente')
        grafico = selecionar_grafico(quantidade)
        
    graficos = os.listdir(output_folder)
    graficos.sort()
    grafico_path = os.path.join(output_folder, graficos[grafico-1])
    return grafico_path

def analise_grafico(grafico_path):
    img = Image.open(grafico_path)

    prompt = """Analise o gráfico e me retorne apenas suas medidas conforme o modelo:
        "x_data = [0, 1, 2, 3, 4, 5]
        y_data = [3, 3, 3, 2, 1, 0]
        x_axis_label_text = "v (m/s)"
        y_axis_label_text = "t (s)" " se houver alguma incógnita, estime o valor dela"""
    response = model.generate_content(
        contents=[prompt,img]
    )
    return response.text
    
def recortarVariaveis(data_string):
    """
    Extrai um bloco de código Python de uma string e executa-o
    para retornar as variáveis definidas dentro do bloco.

    Args:
        data_string (str): A string contendo o bloco de código a ser extraído.

    Returns:
        dict: Um dicionário contendo as variáveis extraídas do bloco de código.
              Retorna um dicionário vazio se o bloco de código não for encontrado.
    """
    # --- Extração do Bloco de Código ---

    # Encontra a posição inicial do bloco de código.
    # Adicionamos 3 para pular os caracteres "```" e começar no código real.
    start_index = data_string.find("```")
    if start_index == -1: # Verifica se o marcador inicial foi encontrado
        print("Erro: Marcador inicial de código '```' não encontrado na string.")
        return {}
    start_index += 3

    # Encontra a posição final do bloco de código.
    # Usamos rfind para garantir que pegamos o último "```".
    end_index = data_string.rfind("```")
    if end_index == -1 or end_index <= start_index: # Verifica se o marcador final foi encontrado e é válido
        print("Erro: Marcador final de código '```' não encontrado ou inválido na string.")
        return {}

    # Extrai a substring que contém apenas o código Python.
    # O método .strip() remove quaisquer espaços em branco extras ou quebras de linha
    # no início e no final do bloco extraído.
    code_block = data_string[start_index:end_index].strip()

    # --- Execução do Código e Captura das Variáveis ---

    # Criamos um dicionário vazio que será usado como o escopo local para exec().
    # Isso fará com que as variáveis definidas no code_block sejam armazenadas aqui.
    extracted_vars = {}
    try:
        # A função exec() executa a string como código Python.
        # Passamos um dicionário vazio para o escopo global (primeiro {})
        # e nosso dicionário 'extracted_vars' para o escopo local (segundo {}).
        # Isso garante que as variáveis do code_block sejam criadas dentro de extracted_vars.
        exec(code_block, {}, extracted_vars)
    except Exception as e:
        print(f"Erro ao executar o bloco de código: {e}")
        return {}

    # Retorna o dicionário contendo todas as variáveis que foram definidas
    # no bloco de código.
    return extracted_vars




def main():
    start()
    question_img = import_img()
    graph_images = cut_image(question_img)
    save(graph_images)
    quantidade = listar_graficos()
    grafico_path = selecionar_grafico(quantidade)
    valores_analise_grafico = analise_grafico(grafico_path)
    print(valores_analise_grafico)
    valores_grafico = recortarVariaveis(valores_analise_grafico)
    grafico = graph_creator.graficoobj(valores_grafico)
    print("gerado")
    print(grafico)


if __name__ == '__main__':
    main()