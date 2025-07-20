print('Iniciando...')
import cv2 # Processamento e manipulação de imagens 
from ultralytics import YOLO # Análisa objetos na imagem com Machine Learnin
import os # Operações com o sistema operacional
from PIL import Image # Prepara imagens para o Gemini
import google.generativeai as genai # Uso temporáio do gemini para detecção dos valores dos gráficos
import graph_creator # Importa .py de criação de graficos 3D

# Tentar carregar variáveis de ambiente do arquivo .env
try:
    from dotenv import load_dotenv # Responsável por ler chave da API do google se estiver em arquivo .env 
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

# Configura pasta de saída de resultados dos gráficos encontrados naa imagem
output_folder = './results/'
# Verifica se a pasta de saída existe, caso contrário, cria
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def clear():
    # Essa função é responsável por fazer a limpiza do terminal com base no sistema eperacional
    if os.name == 'nt':  # 'nt' é para Windows
        os.system('cls')
    else:  # 'posix' é para sistemas Unix/Linux (incluindo macOS)
        os.system('clear')

def start():
    # Essa função é responsável por mostrar o Título da aplicação
    clear()
    print("""\n Bem vindo à
                 𝔼𝕕𝕦𝕍𝕚𝕤𝕚𝕠𝕟   𝕀𝔸 \n""")

def menu():
    # Essa função é responsável de mostrar a tela principal com seus elementos de título e menu
    clear()
    start()
    print('''Escolha uma opção:
             1. Análisar uma imagem
             2. Listar gráficos
             3. Criar modelo de um Gráfico da lista
             4. Imprimir gráfico
             5. Encerrar o programa
''')
    return 

def import_img():
    '''
    Essa função é responsável por verificar e importar a imagem para análise.
    Inputs: Caminho da imagem.
    Output: - Imagem
            - Nome_do_arquivo.
    '''
    
    image_path = input('Diga o caminho da imagem: ')
    if (os.path.exists(image_path)):
        image = cv2.imread(image_path)
        nome_arquivo = os.path.basename(image_path)  # Extrai apenas o nome do arquivo
        nome_arquivo = os.path.splitext(nome_arquivo)[0]  # Retmove a extenção do arquivo ".png"
        return image, nome_arquivo
    else:
        start()
        print('Caminho não existe! \n')
        return import_img()

def cut_image(image, nome_arquivo):
    '''
    Essa função é responsável por buscar os elementos(Enuniados e Gráficos) na imagem e fazer o recorte dos gráficos.
    Inputs: - Imagem
            - Nome da imagem.
    Output: Salva os recortes dos gráficos.
    '''
    print('Buscando gráficos na imagem... \n')
    model = YOLO("./MLs/ML1.pt")
    results = model(image) # Faz a analise com a ML e armazena os resultados em results
    result = results[0] # Define como result, os retornos com a tag 0, ou seja a tag de gráfico, enquanto a tag 1 são os enunciados
    graph_counter = 0 
    # Faz o recorte dos gráficos
    for i, box in enumerate(result.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        if label == 'grafico':
            conf = float(box.conf[0])
            # Recorte do objeto
            cropped = image[y1:y2, x1:x2]
            # Criar nome de arquivo de saída
            output_path = os.path.join(output_folder, f"{nome_arquivo}-Grafico_{graph_counter+1}.jpg")
            # Salvar recorte
            cv2.imwrite(output_path, cropped)
            graph_counter += 1
    print(f"* {graph_counter} gráficos foram encontrados e salvos em {output_folder}")
    input('Pressione entar para continuar...')
    return main()

def deletar_grafico():
    start()
    listar_graficos()
    excluir = input('Deseja excluir algum gráfico? (s/n):').lower()
    if  excluir == 's' or excluir == 'sim':
        grafico_path, _ = selecionar_grafico()
        if graph_path != None:
            os.remove(grafico_path)
            print('Gráfico removido com sucesso.')
        return deletar_grafico()
    elif excluir == 'n' or excluir =='não':
        main()
    else:
        input('Opção inválida, pressione enter para tente novamente')
        return deletar_grafico()
    
        
def listar_graficos():
    '''
    Essa função é responsável por mostrar a lista de imagens de gráficos e deletar se necessário.
    Inputs: - 'deseja apagar algum gráfico?'
            - gráfico para ser deletado.
    Output: - Deleta o gráfico solicitado
    '''

    print('Gráficos:\n')
    graficos = os.listdir(output_folder)
    graficos.sort()
    count = 0
    for i in graficos:
        count = count+1
        print(f'{count}. {i}')
    if not count:
        print(f'*{'Nenhum gráfico na lista :c'.center(30)}*\n')
        input('Pressione enter para voltar para o menu principal...')
        main()
    print()
def selecionar_grafico():
    '''
    Essa função é responsável por fazer a seleção do gráfico, pode ser usado em qualquer circunstancia.
    Input: - gráfico que para selecionar
    Output: - Retorna o caminho da imagem selecionada.
    '''
    #Conta a quantidade de imagens de gráficos na pasta de saida
    graficos = os.listdir(output_folder)
    graficos.sort()
    quantidade = 0
    for i in graficos:
        quantidade = quantidade+1
    try:    
        grafico = int(input('Digite o número do gráfico para selecionar: '))
        if grafico >0 and grafico <= quantidade:
            clear()
            print(f'Gráfico {grafico} selecionado')
        else:
            input('Gráfico não encontrado, pressione enter para tentar novamente')
            return None, None
    except:
        input('Digite o número do gráfico desejado, pressione enter para tentar novamente')
        return None, None
        
    graficos = os.listdir(output_folder)
    graficos.sort()
    grafico_path = os.path.join(output_folder, graficos[grafico-1])
    # Extrai o nome do arquivo
    nome_arquivo = os.path.basename(grafico_path)  # Extrai apenas o nome do arquivo
    nome_arquivo = os.path.splitext(nome_arquivo)[0]  # Retmove a extenção do arquivo ".png"
    return grafico_path, nome_arquivo

def analise_grafico(grafico_path):
    '''
    Essa função é responsável por fazer a análise do gráfico com o gemini(temporáriamente)
    Input: - Caminho do gráfico para análise
    Output: - Retornar os dados do gráfico enviado
    '''
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
    Essa função é responsável por recortar os valores do gráfico retornados pelo gemini
    Inputs: - data_string (str): A string contendo o bloco de código a ser extraído.
    Outputs: - dict: Um dicionário contendo as variáveis extraídas do bloco de código.
             - Retorna um dicionário vazio se o bloco de código não for encontrado.
    """
    # --- Extração do Bloco de Código ---

    # Encontra a posição inicial do bloco de código.
    start_index = data_string.find("```")
    if start_index == -1: # Verifica se o marcador inicial foi encontrado
        print("Erro: Marcador inicial de código '```' não encontrado na string.")
        return
    # Adicionamos 3 para pular os caracteres "```" e começar no código real.
    start_index += 3

    # Encontra a posição final do bloco de código.
    # Usamos rfind para garantir que pegamos o último "```".
    end_index = data_string.rfind("```")
    if end_index == -1 or end_index <= start_index: # Verifica se o marcador final foi encontrado e é válido
        print("Erro: Marcador final de código '```' não encontrado ou inválido na string.")
        return 

    # Extrai a substring que contém apenas o código Python.
    # O método .strip() remove quaisquer espaços em branco extras ou quebras de linha no início e no final do bloco extraído.
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

def criar_modelo3d():
    start()
    listar_graficos()
    graph_path, graph_name = selecionar_grafico()
    if graph_path != None:
        graph_values = analise_grafico(graph_path)
        graph_values_dict = recortarVariaveis(graph_values)
        graph_creator.graficoobj(graph_values_dict,graph_name)
        input('Pressione enter para voltar para o menu principal...')
        main()
    else:
        criar_modelo3d()

def main():
    opcao = menu() # Mostra o menu da aplicação
    opcao = int(input('Digite o número da opção e pressione enter: '))
    try:
        match opcao:
            case 1: # 1. Análisar uma imagem
                start()
                image, nome_arquivo = import_img()
                cut_image(image, nome_arquivo)
                
            case 2: # 2. Listar gráficos
                deletar_grafico()
            case 3: # 3. Criar modelo de um Gráfico da lista
                criar_modelo3d()
            case 4: # 4. Imprimir gráfico
                input('\nFunção em desenvolvimento...')
                return main()
            case 5: # 5. Encerrar o programa
                clear()
                print('Finalizando programa...')
                exit
            case _:
                input('Opção invalida, digite uma opção válida! Pressione enter para tentar novamente...')
                main()
    except:
        input('Opção invalida, digite um número! Pressione enter para tentar novamente...')
        main()

if __name__ == '__main__':
    main()