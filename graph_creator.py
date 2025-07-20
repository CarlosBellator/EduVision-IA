import numpy as np # Computação numérica e arrays
from shapely.geometry import LineString, Point, Polygon
import trimesh # Manipulação de objetos 3D 
import trimesh.transformations as tf # Manipulação de objetos 3D




def graficoobj(valores_grafico,graph_name,output_folder):
    # ------------------------
    # CONFIGURAÇÕES GLOBAIS E DO GRÁFICO
    # ------------------------
    # Dados do gráfico (exemplo)
    x_data = [0, 1, 2, 3, 4, 5]
    y_data = [2, 3, 4, 4, 3, 2]
    x_axis_label_text = "t (s)"
    y_axis_label_text = "v (m/s)"

    x_data = valores_grafico.get("x_data")
    y_data = valores_grafico.get("y_data")
    # Nomes dos eixos (para referência e para os rótulos em Braille)

    x_axis_label_text = valores_grafico.get("x_axis_label_text")
    y_axis_label_text = valores_grafico.get("x_axis_label_text")

    # Parâmetros de design (ajuste conforme necessário)
    altura_base_plataforma = 0.14       # Altura (espessura) da base principal
    margem_base = 0.5                  # Margem ao redor de TODO o conteúdo na base

    altura_linha_grafico = 0.3         # Altura (espessura) da linha do gráfico extrudada
    largura_linha_grafico = 0.15       # Largura da faixa da linha do gráfico

    # Parâmetros para as "meia bolinhas" (relevos) na linha do gráfico
    diametro_relevo_linha = 0.14        # Diâmetro da base da "bolinha" de relevo
    altura_relevo_linha = diametro_relevo_linha / 2 # Altura da "bolinha" (hemisfério)
    espacamento_relevos_linha = 0.4    # Espaçamento entre os centros dos relevos na linha

    # Parâmetros da grade
    espessura_grade = 0.05             # Espessura das linhas da grade
    altura_grade = 0.15                # Altura das linhas da grade em relação à base
    espacamento_grade_x = 1.0          # Espaçamento da grade no eixo X
    espacamento_grade_y = 1.0          # Espaçamento da grade no eixo Y

    # Parâmetros dos eixos (linhas principais)
    espessura_eixo = 0.08              # Espessura das linhas dos eixos X e Y
    altura_eixo = 0.18                 # Altura das linhas dos eixos em relação à base

    # Parâmetros para os textos/números em Braille (simulado com pontos)
    altura_ponto_braille = 0.13        # Altura do relevo de cada ponto Braille
    raio_ponto_braille = 0.04          # Raio de cada ponto Braille
    espacamento_ponto_braille_x = 0.22 # Espaçamento horizontal entre centros de colunas de pontos em uma célula Braille
    espacamento_ponto_braille_y = 0.22 # Espaçamento vertical entre centros de linhas de pontos em uma célula Braille
    espacamento_char_braille = 0.32     # Espaçamento horizontal entre células Braille adjacentes

    # Deslocamento dos rótulos numéricos Braille em relação aos eixos
    offset_rotulos_x_braille = -0.3    # Deslocamento vertical da linha de base dos rótulos do eixo X (negativo = abaixo)
    offset_rotulos_y_braille = -0.3    # Deslocamento horizontal da borda direita dos rótulos do eixo Y (negativo = à esquerda)

    # Deslocamento para os rótulos das unidades dos eixos
    padding_x_axis_unit_label = 0.5    # Espaço após o final da área do gráfico para o rótulo da unidade X
    padding_y_axis_unit_label = 0.3    # Espaço acima do topo da área do gráfico para o rótulo da unidade Y


    # Definição dos padrões Braille (UEB Grade 1 para letras e números, alguns símbolos comuns)
    # Coordenadas dos pontos: [coluna, linha], onde coluna 0=esquerda, 1=direita; linha 0=inferior, 1=meio, 2=superior
    BRAILLE_PATTERNS = {
        # Números (precedidos por sinal de número ⠼ em Braille formal, omitido aqui por simplicidade)
        '0': [[0,1], [1,2], [1,1]],         # ⠚ (dots 2,4,5) - Padrão para 'j', frequentemente usado para 0 em contextos simples. Standard UEB 0 é igual a j.
        '1': [[0,2]],                       # ⠁ (dot 1)
        '2': [[0,2], [0,1]],                 # ⠃ (dots 1,2)
        '3': [[0,2], [1,2]],                 # ⠉ (dots 1,4)
        '4': [[0,2], [1,2], [1,1]],         # ⠙ (dots 1,4,5)
        '5': [[0,2], [1,1]],                 # ⠑ (dots 1,5)
        '6': [[0,2], [0,1], [1,2]],         # ⠋ (dots 1,2,4)
        '7': [[0,2], [0,1], [1,2], [1,1]], # ⠛ (dots 1,2,4,5)
        '8': [[0,2], [0,1], [1,1]],         # ⠓ (dots 1,2,5)
        '9': [[0,1], [1,2]],                 # ⠊ (dots 2,4)
        # Letras minúsculas
        'a': [[0,2]], 'b': [[0,2], [0,1]], 'c': [[0,2], [1,2]], 'd': [[0,2], [1,2], [1,1]],
        'e': [[0,2], [1,1]], 'f': [[0,2], [0,1], [1,2]], 'g': [[0,2], [0,1], [1,2], [1,1]],
        'h': [[0,2], [0,1], [1,1]], 'i': [[0,1], [1,2]], 'j': [[0,1], [1,2], [1,1]],
        'k': [[0,2], [0,0]], 'l': [[0,2], [0,1], [0,0]], 'm': [[0,2], [0,0], [1,2]],
        'n': [[0,2], [0,0], [1,2], [1,1]], 'o': [[0,2], [0,0], [1,1]],
        'p': [[0,2], [0,1], [0,0], [1,2]], 'q': [[0,2], [0,1], [0,0], [1,2], [1,1]],
        'r': [[0,2], [0,1], [0,0], [1,1]], 's': [[0,1], [0,0], [1,2]],
        't': [[0,1], [0,0], [1,2], [1,1]], 'u': [[0,2], [0,0], [1,0]],
        'v': [[0,2], [0,1], [0,0], [1,0]], 'w': [[0,1], [1,2], [1,1], [1,0]],
        'x': [[0,2], [0,0], [1,2], [1,0]], 'y': [[0,2], [0,0], [1,2], [1,1], [1,0]],
        'z': [[0,2], [0,0], [1,1], [1,0]],
        # Símbolos
        '(': [[0,2], [0,1], [1,0]],         # ⠣ (dots 1,2,6 - UEB opening parenthesis)
        ')': [[0,0], [1,2], [1,1]],         # ⠜ (dots 3,4,5 - UEB closing parenthesis)
        '/': [[0,0], [1,2]],                 # ⠌ (dots 3,4 - UEB slash)
        '.': [[0,1], [1,1], [0,0]],         # ⠲ (dots 2,5,6 - UEB period)
        ',': [[0,1]],                       # ⠂ (dot 2 - UEB comma)
        '-': [[0,0], [1,0]],                 # ⠤ (dots 3,6 - UEB hyphen)
    }
    # Altura total de uma célula Braille (3 linhas de pontos)
    ALTURA_CELULA_BRAILLE_COMPLETA = 2 * espacamento_ponto_braille_y + 2 * raio_ponto_braille
    # Largura base de uma célula Braille (2 colunas de pontos)
    LARGURA_CELULA_BRAILLE_BASE = espacamento_ponto_braille_x + 2 * raio_ponto_braille


    # ------------------------
    # FUNÇÕES AUXILIARES
    # ------------------------

    def criar_ponto_braille(posicao_centro, altura_ponto, raio_ponto, z_base):
        """Cria um pequeno cilindro para representar um ponto Braille."""
        ponto = trimesh.creation.cylinder(radius=raio_ponto, height=altura_ponto, sections=8) # sections para suavidade
        ponto.apply_translation([posicao_centro[0], posicao_centro[1], z_base + altura_ponto / 2])
        return ponto

    def _calcular_largura_texto_braille(texto_str):
        """Calcula a largura total que um texto em Braille ocupará."""
        if not texto_str:
            return 0
        
        largura_total = 0
        primeiro_char_processado = False
        for char_atual in texto_str.lower():
            if primeiro_char_processado:
                largura_total += espacamento_char_braille
            
            largura_total += LARGURA_CELULA_BRAILLE_BASE
            primeiro_char_processado = True
                
        return largura_total

    def gerar_malhas_braille_para_texto(texto_str, posicao_texto_canto_inferior_esquerdo_xy, z_base_texto):
        """
        Gera malhas 3D para uma string de texto em Braille.
        posicao_texto_canto_inferior_esquerdo_xy: Coordenadas (x,y) do canto inferior esquerdo da primeira célula Braille.
        z_base_texto: Altura Z da base onde o Braille será colocado.
        Retorna uma lista de malhas (pontos).
        """
        malhas_texto_braille = []
        cursor_x = posicao_texto_canto_inferior_esquerdo_xy[0]
        base_y = posicao_texto_canto_inferior_esquerdo_xy[1]

        primeiro_char_processado = False
        for char_atual in texto_str.lower(): 
            if primeiro_char_processado:
                cursor_x += espacamento_char_braille 
            
            if char_atual == ' ':
                cursor_x += LARGURA_CELULA_BRAILLE_BASE
                primeiro_char_processado = True
                continue

            if char_atual in BRAILLE_PATTERNS:
                padrao_pontos = BRAILLE_PATTERNS[char_atual]
                for col_idx, linha_idx in padrao_pontos:
                    x_ponto_abs = cursor_x + (col_idx * espacamento_ponto_braille_x) + raio_ponto_braille 
                    y_ponto_abs = base_y + (linha_idx * espacamento_ponto_braille_y) + raio_ponto_braille
                    
                    ponto_mesh = criar_ponto_braille(
                        (x_ponto_abs, y_ponto_abs),
                        altura_ponto_braille,
                        raio_ponto_braille,
                        z_base_texto
                    )
                    malhas_texto_braille.append(ponto_mesh)
                
                cursor_x += LARGURA_CELULA_BRAILLE_BASE 
                primeiro_char_processado = True
            else:
                print(f"Aviso: Caractere Braille para '{char_atual}' não definido. Tratando como espaço.")
                cursor_x += LARGURA_CELULA_BRAILLE_BASE 
                primeiro_char_processado = True
                
        return malhas_texto_braille

    # ------------------------
    # 1. PREPARAÇÃO DOS DADOS E COORDENADAS INICIAIS
    # ------------------------
    # Lista para armazenar todas as malhas que ficarão sobre a base
    malhas_elementos_sobre_base = []
    z_superficie_base = altura_base_plataforma # Z onde a superfície da base termina e os elementos começam

    pontos_np = np.column_stack((x_data, y_data))
    linha_principal_2d = LineString(pontos_np)

    min_x_dados, min_y_dados = np.min(pontos_np, axis=0)
    max_x_dados, max_y_dados = np.max(pontos_np, axis=0)

    # O offset_global e area_util definem o posicionamento do *gráfico* em si.
    # A margem_base será usada *depois* para criar a plataforma ao redor de tudo.
    # Usamos uma "margem interna" conceitual para posicionar o gráfico.
    # Para simplificar, vamos usar a `margem_base` como essa margem interna inicial,
    # sabendo que a base final usará `margem_base` novamente para o contorno externo.
    margem_interna_grafico = margem_base

    offset_x_global = margem_interna_grafico - min_x_dados
    offset_y_global = margem_interna_grafico - min_y_dados

    pontos_transformados_np = pontos_np + np.array([offset_x_global, offset_y_global])
    linha_principal_transformada_2d = LineString(pontos_transformados_np)

    area_util_x_min = offset_x_global + min_x_dados
    area_util_x_max = offset_x_global + max_x_dados
    area_util_y_min = offset_y_global + min_y_dados
    area_util_y_max = offset_y_global + max_y_dados

    # ------------------------
    # 2. LINHA DO GRÁFICO COM RELEVOS (MEIA BOLINHAS)
    # ------------------------
    faixa_2d_linha = linha_principal_transformada_2d.buffer(largura_linha_grafico / 2, cap_style='flat', join_style='round')
    if not faixa_2d_linha.is_empty and isinstance(faixa_2d_linha, Polygon):
        linha_grafico_3d = trimesh.creation.extrude_polygon(faixa_2d_linha, altura_linha_grafico)
        linha_grafico_3d.apply_translation([0, 0, z_superficie_base])
        malhas_elementos_sobre_base.append(linha_grafico_3d)

        raio_relevo = diametro_relevo_linha / 2
        comprimento_total_linha = linha_principal_transformada_2d.length
        if espacamento_relevos_linha > 0:
            num_relevos = int(comprimento_total_linha // espacamento_relevos_linha)
            for i in range(num_relevos + 1):
                ponto_na_linha = linha_principal_transformada_2d.interpolate(i * espacamento_relevos_linha)
                x_relevo, y_relevo = ponto_na_linha.x, ponto_na_linha.y
                relevo_esfera = trimesh.creation.icosphere(subdivisions=2, radius=raio_relevo)
                z_pos_relevo = z_superficie_base + altura_linha_grafico + altura_relevo_linha - raio_relevo
                relevo_esfera.apply_translation([x_relevo, y_relevo, z_pos_relevo])
                malhas_elementos_sobre_base.append(relevo_esfera)
    else:
        print("Aviso: Faixa 2D da linha do gráfico está vazia ou inválida. Linha não será criada.")


    # ------------------------
    # 3. GRADE E EIXOS
    # ------------------------
    linha_eixo_x_2d = LineString([(area_util_x_min, area_util_y_min), (area_util_x_max, area_util_y_min)])
    eixo_x_poly = linha_eixo_x_2d.buffer(espessura_eixo / 2, cap_style='flat')
    if not eixo_x_poly.is_empty and isinstance(eixo_x_poly, Polygon):
        eixo_x_3d = trimesh.creation.extrude_polygon(eixo_x_poly, altura_eixo)
        eixo_x_3d.apply_translation([0, 0, z_superficie_base])
        malhas_elementos_sobre_base.append(eixo_x_3d)

    linha_eixo_y_2d = LineString([(area_util_x_min, area_util_y_min), (area_util_x_min, area_util_y_max)])
    eixo_y_poly = linha_eixo_y_2d.buffer(espessura_eixo / 2, cap_style='flat')
    if not eixo_y_poly.is_empty and isinstance(eixo_y_poly, Polygon):
        eixo_y_3d = trimesh.creation.extrude_polygon(eixo_y_poly, altura_eixo)
        eixo_y_3d.apply_translation([0, 0, z_superficie_base])
        malhas_elementos_sobre_base.append(eixo_y_3d)

    valores_x_grade = np.arange(min_x_dados, max_x_dados + espacamento_grade_x / 2, espacamento_grade_x)
    for gx_dado in valores_x_grade:
        gx_base = gx_dado + offset_x_global
        if gx_base > area_util_x_min + espessura_eixo/2 and gx_base < area_util_x_max - espessura_eixo/2 : # Pequena folga
            linha_v_2d = LineString([(gx_base, area_util_y_min), (gx_base, area_util_y_max)])
            grade_v_poly = linha_v_2d.buffer(espessura_grade / 2, cap_style='flat')
            if not grade_v_poly.is_empty and isinstance(grade_v_poly, Polygon):
                grade_v_3d = trimesh.creation.extrude_polygon(grade_v_poly, altura_grade)
                grade_v_3d.apply_translation([0, 0, z_superficie_base])
                malhas_elementos_sobre_base.append(grade_v_3d)

    valores_y_grade = np.arange(min_y_dados, max_y_dados + espacamento_grade_y / 2, espacamento_grade_y)
    for gy_dado in valores_y_grade:
        gy_base = gy_dado + offset_y_global
        if gy_base > area_util_y_min + espessura_eixo/2 and gy_base < area_util_y_max - espessura_eixo/2: # Pequena folga
            linha_h_2d = LineString([(area_util_x_min, gy_base), (area_util_x_max, gy_base)])
            grade_h_poly = linha_h_2d.buffer(espessura_grade / 2, cap_style='flat')
            if not grade_h_poly.is_empty and isinstance(grade_h_poly, Polygon):
                grade_h_3d = trimesh.creation.extrude_polygon(grade_h_poly, altura_grade)
                grade_h_3d.apply_translation([0, 0, z_superficie_base])
                malhas_elementos_sobre_base.append(grade_h_3d)

    # ------------------------
    # 4. RÓTULOS DOS EIXOS EM BRAILLE (NÚMEROS E UNIDADES)
    # ------------------------
    y_baseline_rotulos_x = area_util_y_min + offset_rotulos_x_braille - ALTURA_CELULA_BRAILLE_COMPLETA
    for val_x_dado in sorted(list(set(x_data))):
        val_x_str = str(int(val_x_dado))
        pos_x_marca_eixo = val_x_dado + offset_x_global    
        largura_num_braille = _calcular_largura_texto_braille(val_x_str)
        x_start_num = pos_x_marca_eixo - (largura_num_braille / 2) 
        malhas_num_x = gerar_malhas_braille_para_texto(val_x_str, (x_start_num, y_baseline_rotulos_x), z_superficie_base)
        malhas_elementos_sobre_base.extend(malhas_num_x)

    x_right_edge_rotulos_y = area_util_x_min + offset_rotulos_y_braille
    for val_y_dado in sorted(list(set(y_data))):
        val_y_str = str(int(val_y_dado))
        pos_y_marca_eixo = val_y_dado + offset_y_global    
        largura_num_braille = _calcular_largura_texto_braille(val_y_str)
        x_start_num = x_right_edge_rotulos_y - largura_num_braille 
        y_start_num = pos_y_marca_eixo - (ALTURA_CELULA_BRAILLE_COMPLETA / 2)    
        malhas_num_y = gerar_malhas_braille_para_texto(val_y_str, (x_start_num, y_start_num), z_superficie_base)
        malhas_elementos_sobre_base.extend(malhas_num_y)

    largura_x_unit_label = _calcular_largura_texto_braille(x_axis_label_text)
    x_start_x_unit_label = area_util_x_max + padding_x_axis_unit_label
    y_start_x_unit_label = y_baseline_rotulos_x 
    malhas_x_unit = gerar_malhas_braille_para_texto(x_axis_label_text, (x_start_x_unit_label, y_start_x_unit_label), z_superficie_base)
    malhas_elementos_sobre_base.extend(malhas_x_unit)

    largura_y_unit_label = _calcular_largura_texto_braille(y_axis_label_text)
    y_start_y_unit_label = area_util_y_max + padding_y_axis_unit_label
    x_start_y_unit_label = x_right_edge_rotulos_y - largura_y_unit_label 
    malhas_y_unit = gerar_malhas_braille_para_texto(y_axis_label_text, (x_start_y_unit_label, y_start_y_unit_label), z_superficie_base)
    malhas_elementos_sobre_base.extend(malhas_y_unit)

    # ------------------------
    # 5. CRIAR BASE SÓLIDA AJUSTADA
    # ------------------------
    malhas_finais = []
    if not malhas_elementos_sobre_base:
        print("Nenhum elemento gráfico foi gerado. A base não será criada.")
    else:
        # Calcular limites de todos os elementos sobre a base
        all_mesh_bounds = [m.bounds for m in malhas_elementos_sobre_base if m is not None and hasattr(m, 'bounds') and m.vertices.shape[0] > 0]
        
        if not all_mesh_bounds:
            print("Nenhum elemento com limites válidos foi encontrado. A base não será criada.")
        else:
            min_coords_all = np.min([b[0] for b in all_mesh_bounds], axis=0)
            max_coords_all = np.max([b[1] for b in all_mesh_bounds], axis=0)

            min_ext_x, min_ext_y, _ = min_coords_all
            max_ext_x, max_ext_y, _ = max_coords_all

            # Dimensões do conteúdo total
            conteudo_largura = max_ext_x - min_ext_x
            conteudo_altura = max_ext_y - min_ext_y

            # Dimensões da base final com margem
            base_final_largura = conteudo_largura + 2 * margem_base
            base_final_altura = conteudo_altura + 2 * margem_base

            # Centro do conteúdo (que será o centro da base)
            base_final_centro_x = min_ext_x + conteudo_largura / 2
            base_final_centro_y = min_ext_y + conteudo_altura / 2
            
            base_solida = trimesh.creation.box(extents=(base_final_largura, base_final_altura, altura_base_plataforma))
            base_solida.apply_translation([base_final_centro_x, base_final_centro_y, altura_base_plataforma / 2])
            
            malhas_finais.append(base_solida)
            malhas_finais.extend(malhas_elementos_sobre_base)

    # ------------------------
    # 6. JUNTAR TUDO E EXPORTAR
    # ------------------------
    if malhas_finais:
        modelo_final_combinado = trimesh.util.concatenate(malhas_finais)
        if not modelo_final_combinado.is_watertight:
            print("Atenção: O modelo combinado pode não ser 'watertight'. Tentando corrigir...")
            trimesh.repair.fill_holes(modelo_final_combinado)
            trimesh.repair.fix_normals(modelo_final_combinado) # Tentar fix_normals também
            modelo_final_combinado.merge_vertices() # Adicionar merge_vertices
            if modelo_final_combinado.is_watertight:
                print("Correção de 'watertight' bem-sucedida.")
            else:
                print("Não foi possível tornar o modelo 'watertight' automaticamente. Verifique o STL.")
        nome_arquivo_saida = f'{output_folder}/{graph_name}_tatil.stl'
        modelo_final_combinado.export(nome_arquivo_saida)
        print(f"Arquivo STL '{nome_arquivo_saida}' gerado com sucesso!")
        return modelo_final_combinado  # Retorna o objeto 3D completo
    else:
        print("Nenhuma malha foi gerada para o modelo final. Verifique as configurações e dados.")
        return None  # Retorna None se falhou