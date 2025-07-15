# 📊 EduVision IA

**Sistema de Análise e Conversão de Gráficos para Objetos 3D Táteis**

Um projeto de TCC que utiliza Inteligência Artificial para detectar gráficos em imagens, analisá-los e convertê-los em objetos 3D táteis (.STL) para auxiliar pessoas com deficiência visual no ensino de matemática e ciências.

## 🎯 Objetivo

O **EduVision IA** tem como objetivo tornar gráficos acessíveis para pessoas com deficiência visual, convertendo-os automaticamente em modelos 3D táteis que podem ser impressos em 3D.

## 🔧 Funcionalidades

- **🔍 Detecção Automática**: Utiliza YOLOv8 para detectar gráficos em imagens
- **🧠 Análise Inteligente**: Emprega Google Gemini AI para extrair dados dos gráficos
- **📐 Conversão 3D**: Gera modelos STL táteis com:
  - Linhas do gráfico em relevo
  - Grade de referência
  - Eixos principais
  - Pontos de dados em Braille
  - Rótulos dos eixos em Braille
- **♿ Acessibilidade**: Focado em tornar conteúdo visual acessível

## 📁 Estrutura do Projeto

```
application/
├── main.py                 # Arquivo principal
├── graficosobj.py         # Geração de objetos 3D
├── requirements.txt       # Dependências
├── grafico_tatil_base.stl # Modelo 3D de exemplo
├── model1/
│   └── best.pt           # Modelo YOLOv8 treinado
├── results/              # Gráficos extraídos
└── image/                # Imagens de exemplo
```

## 🚀 Instalação

### Pré-requisitos

- Python 3.8 ou superior
- GPU NVIDIA (opcional, mas recomendado para melhor performance)

### Instalação das Dependências

```bash
# Clone o repositório
git clone https://github.com/CarlosBellator/EduVision-IA.git
cd EduVision-IA/application

# Crie um ambiente virtual
python -m venv venv

# Ative o ambiente virtual
# Linux/macOS:
source venv/bin/activate
# Windows:
#venv\Scripts\activate

# Instale as dependências
pip install -r requirements.txt
```

### Configuração da API Google

1. **Obter API Key**:
   - Acesse [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Faça login com sua conta Google
   - Clique em "Create API Key"
   - Copie a chave gerada

2. **Configurar Variável de Ambiente**:
   
   **Linux/macOS:**
   ```bash
   export GOOGLE_API_KEY="sua_chave_api_aqui"
   ```
   
   **Windows:**
   ```cmd
   set GOOGLE_API_KEY=sua_chave_api_aqui
   ```
   
   **Ou crie um arquivo `.env`:**
   ```bash
   # Copie o arquivo de exemplo
   cp .env.example .env
   
   # Edite o arquivo .env e adicione sua chave
   GOOGLE_API_KEY=sua_chave_api_aqui
   ```

3. **Instalar python-dotenv (se usar arquivo .env)**:
   ```bash
   pip install python-dotenv
   ```

## 💻 Como Usar

### Execução Básica

```bash
python main.py
```

### Fluxo de Trabalho

1. **Importar Imagem**: Forneça o caminho da imagem contendo gráficos
2. **Detecção**: O sistema detecta automaticamente gráficos na imagem
3. **Seleção**: Escolha qual gráfico analisar
4. **Análise**: A IA extrai dados do gráfico selecionado
5. **Conversão**: Gera modelo 3D tátil em formato STL

## 🛠️ Tecnologias Utilizadas

- **🐍 Python**: Linguagem principal
- **👁️ OpenCV**: Processamento de imagens
- **🎯 YOLOv8 (Ultralytics)**: Detecção de objetos
- **🧠 Google Gemini AI**: Análise de gráficos
- **📐 NumPy**: Computação numérica
- **🔺 Shapely**: Geometria computacional
- **🎨 Trimesh**: Manipulação de malhas 3D
- **🔻 Triangle**: Triangulação

## 📊 Compatibilidade

### Sistemas Operacionais
- ✅ Windows
- ✅ Linux
- ✅ macOS

### Hardware
- ✅ **CPU**: Funciona em qualquer processador moderno
- ✅ **GPU**: Acelera processamento (NVIDIA CUDA recomendado)
- ✅ **RAM**: Mínimo 4GB, recomendado 8GB+

## 🔧 Configurações Avançadas

### Parâmetros do Modelo 3D

O arquivo `graficosobj.py` permite ajustar:

```python
# Dimensões da base
altura_base_plataforma = 0.14
margem_base = 0.5

# Linha do gráfico
altura_linha_grafico = 0.3
largura_linha_grafico = 0.15

# Relevos táteis
diametro_relevo_linha = 0.14
altura_relevo_linha = 0.07
espacamento_relevos_linha = 0.4

# Grade de referência
espessura_grade = 0.05
altura_grade = 0.15
espacamento_grade_x = 1.0
espacamento_grade_y = 1.0
```

## 📋 Requisitos do Sistema

### Mínimo
- **OS**: Windows 10/Linux/macOS
- **RAM**: 4GB
- **Python**: 3.8+
- **Espaço**: 2GB livres

### Recomendado
- **OS**: Windows 11/Ubuntu 20.04+/macOS Big Sur+
- **RAM**: 8GB+
- **GPU**: NVIDIA GTX 1060 ou superior
- **Python**: 3.10+
- **Espaço**: 5GB livres

## 🐛 Solução de Problemas

### Erros Comuns

1. **Erro de importação do OpenCV**:
   ```bash
   pip install opencv-python --upgrade
   ```

2. **Modelo YOLO não encontrado**:
   - Verifique se `model1/best.pt` existe
   - Retreine o modelo se necessário

3. **Erro da API Google**:
   - Verifique se a chave da API está correta
   - Confirme se a API está habilitada

## 👨‍💻 Autor

**Carlos Bellator**
- GitHub: [@CarlosBellator](https://github.com/CarlosBellator)
- Projeto: [EduVision-IA](https://github.com/CarlosBellator/EduVision-IA)

---


<div align="center">
  <p><strong>EduVision IA - Tornando gráficos acessíveis através da tecnologia</strong></p>
  <p>Desenvolvido como Trabalho de Conclusão de Curso (TCC)</p>
</div>
