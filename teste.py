import streamlit as st
import control as ct
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

def poly_to_latex(coeffs):
    degree = len(coeffs) - 1
    terms = []
    for i, c in enumerate(coeffs):
        if c == 0: continue
        c_val = float(c)
        if c_val == 1 and i < degree: c_str = ""
        elif c_val == -1 and i < degree: c_str = "-"
        else: c_str = f"{c_val:.1f}"
            
        power = degree - i
        if power == 0: s_str = ""
        elif power == 1: s_str = "s"
        else: s_str = f"s^{power}"
        terms.append(f"{c_str}{s_str}")
        
    if not terms: return "0.0"
    res = terms[0]
    for term in terms[1:]:
        if term.startswith("-"): res += f" - {term[1:]}"
        else: res += f" + {term}"
    return res

def format_factor(roots):
    if len(roots) == 0: return "1.0"
    
    grouped_roots = {}
    for r in roots:
        found = False
        for gr in grouped_roots.keys():
            if np.isclose(r, gr, atol=1e-4):
                grouped_roots[gr] += 1
                found = True
                break
        if not found:
            grouped_roots[r] = 1

    factors = []
    has_isolated_s = False
    s_power = 0

    for r, count in grouped_roots.items():
        power_str = f"^{{{count}}}" if count > 1 else ""
        if np.isreal(r):
            val = float(np.real(r))
            if abs(val) < 1e-4: 
                has_isolated_s = True
                s_power = count
            elif val < 0: 
                factors.append(f"(s + {abs(val):.1f}){power_str}")
            else: 
                factors.append(f"(s - {val:.1f}){power_str}")
        else:
            a = float(np.real(r))
            b = float(np.imag(r))
            str_a = f"+ {-a:.1f}" if a < 0 else f"- {a:.1f}" if a > 0 else ""
            str_b = f"- {b:.1f}i" if b > 0 else f"+ {-b:.1f}i"
            factors.append(f"(s {str_a} {str_b}){power_str}")
    
    result = "".join(factors)
    if has_isolated_s:
        s_str = "s" if s_power == 1 else f"s^{{{s_power}}}"
        result = s_str + result
    return result


st.set_page_config(page_title="Calculadora LGR", layout="centered")

# --- CSS PERSONALIZADO PARA MUDAR A COR DA BARRA LATERAL ---
st.markdown(
    """
    <style>
    /* Altera a cor de fundo da barra lateral para rosa malva */
    [data-testid="stSidebar"] {
        background-color: #c54b8c; 
    }
    
    /* Força todos os textos, títulos, labels, parágrafos e spans da barra lateral a ficarem em branco */
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] label, 
    [data-testid="stSidebar"] span, 
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] div {
        color: #ffffff !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Calculadora do Lugar Geométrico de Raízes")
st.markdown("""
**Aluno:** Karen Helloisa Araújo <br>
**Curso:** Engenharia Mecatrônica — UFRN/2026.2.<br>
**Disciplina:** Projeto de Sistemas de Controle<br>
**Projeto:** Calculadora Analítica de Lugar Geométrico das Raízes (LGR)
""", unsafe_allow_html=True)

st.markdown("""
        
Esta aplicação foi desenvolvida para auxiliar estudantes e engenheiros na análise detalhada de sistemas de controle. 
O **Lugar das Raízes** é uma ferramenta gráfica fundamental que mostra como os pólos de um sistema em malha fechada 
se movimentam no plano complexo $s$ à medida que o ganho do controlador ($K$) varia de zero a infinito. Através deste sistema, você pode:
* Inserir as funções de transferência de malha direta $G(s)$ e realimentação $H(s)$ na barra lateral.
* Acompanhar a resolução passo a passo detalhada (cálculo de pólos, zeros, assíntotas, pontos de fuga e cruzamentos).
* Testar a pertinência de pontos específicos ($s_i$) no plano complexo por meio do **critério do ângulo de fase** e determinar o ganho $K$ correspondente.
""")


# --- PAINEL DE CONTROLE LATERAL (Parâmetros e Teste de Pontos) ---
st.sidebar.header("⚙️ Parâmetros do Sistema")
st.sidebar.markdown("Insira os coeficientes do maior para o menor grau:")

g_num_str = st.sidebar.text_input("G(s) - Numerador")
g_den_str = st.sidebar.text_input("G(s) - Denominador")
h_num_str = st.sidebar.text_input("H(s) - Numerador")
h_den_str = st.sidebar.text_input("H(s) - Denominador",)


st.sidebar.header("🎯 Teste de Ponto no LGR")
si_real = st.sidebar.number_input("Parte real do ponto de teste ($s_i$):", value=None, format="%.2f")
si_imag = st.sidebar.number_input("Parte imaginária do ponto de teste ($s_i$):", value=None, format="%.2f")
tolerancia = st.sidebar.number_input("Tolerância (graus):", value=None, format="%.2f")

gerar_btn = st.sidebar.button("Gerar Resolução Completa", type="primary", use_container_width=True)

if gerar_btn:
    try:
        g_num = [float(x) for x in g_num_str.split()]
        g_den = [float(x) for x in g_den_str.split()]
        h_num = [float(x) for x in h_num_str.split()]
        h_den = [float(x) for x in h_den_str.split()]
        
        sys_G = ct.TransferFunction(g_num, g_den)
        sys_H = ct.TransferFunction(h_num, h_den)
        sys_GH = sys_G * sys_H
        
        num = sys_GH.num[0][0].tolist()
        den = sys_GH.den[0][0].tolist()
        
        sys = ct.TransferFunction(num, den)
        polos = sys.poles()
        zeros = sys.zeros()
        np_count = len(polos)
        nz_count = len(zeros)
        
        st.markdown("---")

        st.subheader("Sistema de Malha Aberta")
        st.write("Funções de transferência inseridas:")
        
        g_latex_num = poly_to_latex(g_num)
        g_latex_den = poly_to_latex(g_den)
        h_latex_num = poly_to_latex(h_num)
        h_latex_den = poly_to_latex(h_den)
        
        if g_latex_den == "1":
            g_expr = rf"G(s) = {g_latex_num}"
        else:
            g_expr = rf"G(s) = \frac{{{g_latex_num}}}{{{g_latex_den}}}"
            
        if h_latex_num == "1" and h_latex_den == "1":
            h_expr = rf"H(s) = 1"
        elif h_latex_den == "1":
            h_expr = rf"H(s) = {h_latex_num}"
        else:
            h_expr = rf"H(s) = \frac{{{h_latex_num}}}{{{h_latex_den}}}"
        
        col_g, col_h = st.columns(2)
        with col_g:
            st.latex(g_expr)
        with col_h:
            st.latex(h_expr)
            
        st.markdown("---")
        # ---------------------------------------------------------
        # PASSO 1 -------------------------------------------------
        st.subheader("1. Escrever o polinômio característico")
        st.markdown("""
        Escrever o polinômio característico de modo que o parâmetro de interesse K apareça claramente. <br>
        """, unsafe_allow_html=True)
        
        num_latex_final = poly_to_latex(num)
        den_latex_final = poly_to_latex(den)
        
        st.latex(rf"1 + G(s)H(s) = 1 + K \frac{{{num_latex_final}}}{{{den_latex_final}}} = 1 + KP(s)")

        # PASSO 2 -------------------------------------------------       
        st.subheader("2. Fatorar o polinômio P(s) em termos dos pólos e zeros")
        num_factored = format_factor(zeros)
        den_factored = format_factor(polos)
        st.latex(rf"P(s) = \frac{{{num_factored}}}{{{den_factored}}}")

        # PASSO 3 -------------------------------------------------
        st.subheader("3. Polos e Zeros")
        st.write("**Polos:**")
        if np_count > 0:
            for idx, p in enumerate(polos, start=1):
                real_p = float(np.real(p))
                imag_p = float(np.imag(p))
                if abs(imag_p) < 1e-4:
                    st.latex(rf"p_{{{idx}}} = {real_p:.4f} + 0.0000j")
                elif imag_p > 0:
                    st.latex(rf"p_{{{idx}}} = {real_p:.4f} + {imag_p:.4f}j")
                else:
                    st.latex(rf"p_{{{idx}}} = {real_p:.4f} - {abs(imag_p):.4f}j")
        else:
            st.write("Não há polos finitos.")
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.write("**Zeros:**")
        if nz_count > 0:
            for idx, z in enumerate(zeros, start=1):
                real_z = float(np.real(z))
                imag_z = float(np.imag(z))
                if abs(imag_z) < 1e-4:
                    st.latex(rf"z_{{{idx}}} = {real_z:.4f} + 0.0000j")
                elif imag_z > 0:
                    st.latex(rf"z_{{{idx}}} = {real_z:.4f} + {imag_z:.4f}j")
                else:
                    st.latex(rf"z_{{{idx}}} = {real_z:.4f} - {abs(imag_z):.4f}j")
        else:
            st.markdown("*Não há zeros finitos.*")

        # PASSO 4 -------------------------------------------------
        st.subheader("4. Assinalar os segmentos do eixo real que são LGR")
        st.markdown("""
        LGR fica à esquerda de um número ímpar de polos e zeros. Ele se inicia nos polos finitos 
        e infinitos e termina nos zeros finitos e infinitos. <br>
        """, unsafe_allow_html=True)
        
        real_roots = sorted([float(np.real(r)) for r in np.concatenate((polos, zeros)) if np.isreal(r)], reverse=True)
        if real_roots:
            for i in range(len(real_roots)):
                start = real_roots[i]
                if i + 1 < len(real_roots):
                    end_str = f"{real_roots[i+1]:.1f}"
                else:
                    end_str = "$-\\infty$"
                
                start_str = f"{start:.1f}"
                count_right = i + 1
                paridade = "nº Ímpar" if count_right % 2 != 0 else "nº Par"
                
                if count_right % 2 != 0:
                    st.markdown(f"- **Entre {start_str} e {end_str}:** Total de {count_right} pólos/zeros ({paridade}) $\\rightarrow$ **Faz parte do LGR**")
                else:
                    st.markdown(f"- **Entre {start_str} e {end_str}:** Total de {count_right} pólos/zeros ({paridade}) $\\rightarrow$ Não faz parte")
        else:
            st.write("Não há pólos ou zeros no eixo real.")

        col_esq, col_centro, col_dir = st.columns([0.5, 5, 0.5])
        with col_centro:
            fig_real, ax_real = plt.subplots(figsize=(9, 5))
            
            all_p_real = [np.real(p) for p in polos]
            all_p_imag = [np.imag(p) for p in polos]
            ax_real.scatter(all_p_real, all_p_imag, marker='x', color='red', s=80, label='Polos', linewidths=2, zorder=5)
            
            if nz_count > 0:
                all_z_real = [np.real(z) for z in zeros]
                all_z_imag = [np.imag(z) for z in zeros]
                ax_real.scatter(all_z_real, all_z_imag, marker='o', facecolors='none', edgecolors='blue', s=80, label='Zeros', linewidths=2, zorder=5)

            if real_roots:
                for i in range(len(real_roots)):
                    count_right = i + 1
                    if count_right % 2 != 0:
                        start = real_roots[i]
                        end = real_roots[i+1] if i+1 < len(real_roots) else min(all_p_real + [np.real(z) for z in zeros]) - 2.0
                        ax_real.plot([start, end], [0, 0], color='blue', linewidth=4, label='LGR Eixo Real' if i == 0 else "", zorder=3)

            ax_real.axhline(0, color='black', lw=1)
            ax_real.axvline(0, color='black', lw=1)
            
            max_imag = max([abs(np.imag(p)) for p in polos] + [abs(np.imag(z)) for z in zeros] + [1.0])
            max_real = max([abs(np.real(p)) for p in polos] + [abs(np.real(z)) for z in zeros] + [1.0])
            ax_real.set_ylim(-max_imag * 1.3, max_imag * 1.3)
            ax_real.set_xlim(-max_real * 1.2, 1.0)
            
            ax_real.grid(True, linestyle='--', alpha=0.5)
            ax_real.set_title("LGR - Segmentos do Eixo Real")
            ax_real.set_xlabel("Eixo Real ($\sigma$)")
            ax_real.set_ylabel("Eixo Imaginário ($j\omega$)")
            ax_real.legend(loc='upper right')
            st.pyplot(fig_real, use_container_width=True)

        # PASSO 5 -------------------------------------------------
        st.subheader("5. Determinar o número de lugares separados (LS)")
        st.markdown("""
                Lugares Separados são os segmentos de curva que compoem o LGR. <br>
                """, unsafe_allow_html=True)
        st.write(f"**$L_s$ = $n_p$ = {max(np_count, nz_count)}**")

        # PASSO 6 -------------------------------------------------
        st.subheader("6. Simetria")
        st.write("O LGR é simétrico em relação ao eixo real.")

       # PASSO 7 -------------------------------------------------
        st.subheader("7. Assíntotas")
        st.markdown("""
        $(n_p - n_z)$ segmentos de um LGR prosseguem em direção aos zeros infinitos ao longo 
        de assíntotas centralizadas em $\\sigma_A$ e com ângulos $\\phi_A$. <br>
        """, unsafe_allow_html=True)
        
        if np_count > nz_count:
            diff_pz = np_count - nz_count
            sum_p_val = np.sum(np.real(polos))
            sum_z_val = np.sum(np.real(zeros)) if nz_count > 0 else 0.0
            centroide = (sum_p_val - sum_z_val) / diff_pz
            
            st.latex(r"\sigma_A = \frac{\sum(-p_j) - \sum(-z_i)}{n_p - n_z}")
            
            sum_p_str = " + ".join([f"({np.real(p):.2f})" for p in polos])
            sum_z_str = " + ".join([f"({np.real(z):.2f})" for z in zeros]) if nz_count > 0 else "0.00"
            st.latex(rf"\sigma_A = \frac{{({sum_p_val:.2f}) - ({sum_z_val:.2f})}}{{{diff_pz}}} = {centroide:.2f}")
            
            st.latex(r"\phi_A = \frac{(2q + 1)}{n_p - n_z} \cdot 180^\circ")
            
            angulos = []
            for q in range(diff_pz):
                angle = ((2*q + 1) * 180) / diff_pz
                angulos.append(angle)
                st.latex(rf"\phi_{{{q}}} = \frac{{(2({q}) + 1)}}{{{diff_pz}}} \cdot 180^\circ = {angle:.1f}^\circ \quad (q = {q})")
                
            col_esq, col_centro, col_dir = st.columns([0.5, 5, 0.5])
            with col_centro:
                fig_asym, ax_asym = plt.subplots(figsize=(9, 5))
                
                all_p_real = [np.real(p) for p in polos]
                all_p_imag = [np.imag(p) for p in polos]
                ax_asym.scatter(all_p_real, all_p_imag, marker='x', color='red', s=80, label='Polos', linewidths=2, zorder=5)
                
                if nz_count > 0:
                    all_z_real = [np.real(z) for z in zeros]
                    all_z_imag = [np.imag(z) for z in zeros]
                    ax_asym.scatter(all_z_real, all_z_imag, marker='o', facecolors='none', edgecolors='blue', s=80, label='Zeros', linewidths=2, zorder=5)

                real_roots = sorted([float(np.real(r)) for r in np.concatenate((polos, zeros)) if np.isreal(r)], reverse=True)
                if real_roots:
                    for i in range(len(real_roots)):
                        count_right = i + 1
                        if count_right % 2 != 0:
                            start = real_roots[i]
                            end = real_roots[i+1] if i+1 < len(real_roots) else min(all_p_real + [np.real(z) for z in zeros]) - 3.0
                            ax_asym.plot([start, end], [0, 0], color='blue', linewidth=4, label='LGR Eixo Real' if i == 0 else "", zorder=3)

                ax_asym.scatter([centroide], [0], marker='D', color='magenta', s=100, label=f'Centroide ($\\sigma_A$ = {centroide:.2f})', zorder=6)

                x_asym = np.linspace(-10, 5, 400)
                for ang in angulos:
                    rad = np.deg2rad(ang)
                    if not np.isclose(np.cos(rad), 0, atol=1e-3):
                        y_asym = np.tan(rad) * (x_asym - centroide)
                        if 90 < ang < 270:
                            mask = x_asym <= centroide
                        else:
                            mask = x_asym >= centroide
                        ax_asym.plot(x_asym[mask], y_asym[mask], color='orange', linestyle='--', linewidth=1.5, zorder=4)
                    else:
                        ax_asym.axvline(centroide, color='orange', linestyle='--', linewidth=1.5, zorder=4)

                from matplotlib.lines import Line2D
                handles, labels = ax_asym.get_legend_handles_labels()
                handles.append(Line2D([0], [0], color='orange', linestyle='--', lw=1.5, label='Assíntotas'))
                ax_asym.legend(handles=handles, loc='upper right')

                ax_asym.axhline(0, color='black', lw=1)
                ax_asym.axvline(0, color='black', lw=1)
                
                max_imag = max([abs(np.imag(p)) for p in polos] + [abs(np.imag(z)) for z in zeros] + [2.0])
                max_real = max([abs(np.real(p)) for p in polos] + [abs(np.real(z)) for z in zeros] + [abs(centroide)] + [2.0])
                ax_asym.set_ylim(-max_imag * 1.5, max_imag * 1.5)
                ax_asym.set_xlim(-max_real * 1.2, 2.0)
                
                ax_asym.grid(True, linestyle='--', alpha=0.5)
                ax_asym.set_title("LGR com Assíntotas")
                ax_asym.set_xlabel("Eixo Real ($\sigma$)")
                ax_asym.set_ylabel("Eixo Imaginário ($j\omega$)")
                st.pyplot(fig_asym, use_container_width=True)
        else:
            st.info("🔹 **Aviso:** Como o número de zeros é maior ou igual ao de pólos, não existem ramos indo para o infinito. O cálculo de assíntotas não é aplicável.")

        # PASSO 8 -------------------------------------------------
        st.subheader("8. Pontos de Saída/Entrada no Eixo Real")
        
        st.markdown("**1º Fazer $K = p(s)$:**")
        st.write("A partir da equação característica, isolamos $K$:")
        
        num_latex_str = poly_to_latex(num)
        den_latex_str = poly_to_latex(den)
        st.latex(rf"K = p(s) = -\frac{{D(s)}}{{N(s)}} = -\frac{{{den_latex_str}}}{{{num_latex_str}}}")
        
        s = sp.Symbol('s')
        N_sym = sum(c * s**(len(num)-1-i) for i, c in enumerate(num))
        D_sym = sum(c * s**(len(den)-1-i) for i, c in enumerate(den))
        
        p_s = - D_sym / N_sym 
        dp_ds = sp.cancel(sp.diff(p_s, s))
        dp_num, dp_den = sp.fraction(dp_ds)
        
        st.markdown("**2º Determinar as raízes de $\\frac{dp(s)}{ds} = 0$:**")
        st.latex(rf"\frac{{dp(s)}}{{ds}} = {sp.latex(dp_num.expand())} = 0")
        
        eq_zeros = sp.simplify(dp_num)
        st.latex(rf"{sp.latex(eq_zeros.expand())} = 0")
        
        raizes_ram = sp.roots(eq_zeros, s)
        
        lista_todas_raizes = []
        for r, mult in raizes_ram.items():
            for _ in range(mult):
                lista_todas_raizes.append(complex(r.evalf()))
                
        str_raizes = []
        for r in lista_todas_raizes:
            if abs(r.imag) < 1e-4:
                str_raizes.append(f"{r.real:.4f}")
            else:
                s_imag = f"+ {r.imag:.4f}j" if r.imag > 0 else f"- {abs(r.imag):.4f}j"
                str_raizes.append(f"{r.real:.4f} {s_imag}")
                
        st.markdown(f"**Todas as raízes calculadas:** $s = [{', '.join(str_raizes)}]$")
        
        def pertence_ao_lgr(val_real, r_roots):
            sorted_r = sorted([float(np.real(x)) for x in r_roots], reverse=True)
            for idx in range(len(sorted_r)):
                start = sorted_r[idx]
                end = sorted_r[idx+1] if idx+1 < len(sorted_r) else -999999.0
                if (idx + 1) % 2 != 0: # Segmento válido
                    if end <= val_real <= start:
                        return True
            return False

        pontos_validos = []
        all_real_nodes = np.concatenate((polos, zeros))
        for r in lista_todas_raizes:
            if abs(r.imag) < 1e-4:
                val_r = float(r.real)
                if pertence_ao_lgr(val_r, all_real_nodes):
                    pontos_validos.append(val_r)
                    
        if pontos_validos:
            p_val_str = ", ".join([f"{p:.4f}" for p in pontos_validos])
            st.markdown(f"**Raízes válidas (pertencem ao LGR no eixo real):** $s = {{{p_val_str}}}$")
        else:
            st.info("🔹 **Aviso:** Nenhuma das raízes reais da derivada pertence a um segmento válido do LGR no eixo real.")

        col_esq, col_centro, col_dir = st.columns([0.5, 5, 0.5])
        with col_centro:
            fig_break, ax_break = plt.subplots(figsize=(9, 5))
            
            # Plota polos e zeros
            all_p_real = [np.real(p) for p in polos]
            all_p_imag = [np.imag(p) for p in polos]
            ax_break.scatter(all_p_real, all_p_imag, marker='x', color='red', s=80, label='Polos', linewidths=2, zorder=5)
            
            if nz_count > 0:
                all_z_real = [np.real(z) for z in zeros]
                all_z_imag = [np.imag(z) for z in zeros]
                ax_break.scatter(all_z_real, all_z_imag, marker='o', facecolors='none', edgecolors='blue', s=80, label='Zeros', linewidths=2, zorder=5)

            real_roots_sorted = sorted([float(np.real(r)) for r in np.concatenate((polos, zeros)) if np.isreal(r)], reverse=True)
            if real_roots_sorted:
                for i in range(len(real_roots_sorted)):
                    count_right = i + 1
                    if count_right % 2 != 0:
                        start = real_roots_sorted[i]
                        end = real_roots_sorted[i+1] if i+1 < len(real_roots_sorted) else min(all_p_real + [np.real(z) for z in zeros]) - 3.0
                        ax_break.plot([start, end], [0, 0], color='blue', linewidth=4, label='LGR Eixo Real' if i == 0 else "", zorder=3)

            # Plota Assíntotas em laranja tracejado cobrindo corretamente toda a direção do ângulo
            if np_count > nz_count:
                diff_pz = np_count - nz_count
                sum_p_val = np.sum(np.real(polos))
                sum_z_val = np.sum(np.real(zeros)) if nz_count > 0 else 0.0
                centroide = (sum_p_val - sum_z_val) / diff_pz
                
                x_asym = np.linspace(-10, 5, 400)
                for q in range(diff_pz):
                    ang = ((2*q + 1) * 180) / diff_pz
                    rad = np.deg2rad(ang)
                    if not np.isclose(np.cos(rad), 0, atol=1e-3):
                        y_asym = np.tan(rad) * (x_asym - centroide)
                        if 90 < ang < 270:
                            mask = x_asym <= centroide
                        else:
                            mask = x_asym >= centroide
                        ax_break.plot(x_asym[mask], y_asym[mask], color='orange', linestyle='--', linewidth=1.5, zorder=4)
                    else:
                        ax_break.axvline(centroide, color='orange', linestyle='--', linewidth=1.5, zorder=4)

            if pontos_validos:
                for pv in pontos_validos:
                    ax_break.scatter([pv], [0], marker='d', color='green', s=100, zorder=7)
                    ax_break.annotate(f"{pv:.2f}", (pv, 0), textcoords="offset points", xytext=(0, 15),
                                      ha='center', fontsize=9, fontweight='bold', color='green',
                                      bbox=dict(boxstyle="square,pad=0.3", fc="white", ec="green", lw=1),
                                      arrowprops=dict(arrowstyle="->", color="magenta", lw=1))

            from matplotlib.lines import Line2D
            handles, labels = ax_break.get_legend_handles_labels()
            if np_count > nz_count:
                handles.append(Line2D([0], [0], color='orange', linestyle='--', lw=1.5, label='Assíntotas'))
            if pontos_validos:
                handles.append(Line2D([0], [0], marker='d', color='w', markerfacecolor='green', markersize=8, label='Pontos de Saída/Entrada'))
            ax_break.legend(handles=handles, loc='upper right')

            ax_break.axhline(0, color='black', lw=1)
            ax_break.axvline(0, color='black', lw=1)
            
            max_imag = max([abs(np.imag(p)) for p in polos] + [abs(np.imag(z)) for z in zeros] + [2.0])
            max_real = max([abs(np.real(p)) for p in polos] + [abs(np.real(z)) for z in zeros] + [2.0])
            ax_break.set_ylim(-max_imag * 1.5, max_imag * 1.5)
            ax_break.set_xlim(-max_real * 1.2, 2.0)
            
            ax_break.grid(True, linestyle='--', alpha=0.5)
            ax_break.set_title("Lugar das Raízes (com Pontos de Descolagem)")
            ax_break.set_xlabel("Eixo Real ($\sigma$)")
            ax_break.set_ylabel("Eixo Imaginário ($j\omega$)")
            st.pyplot(fig_break, use_container_width=True)

        # PASSO 9 -------------------------------------------------
        st.subheader("9. Cruzamento com o Eixo Imaginário (Routh-Hurwitz)")
        st.markdown(r"Equação Característica $1 + k \frac{N(s)}{D(s)} = 0 \implies D(s) + kN(s) = 0$")
        
        K = sp.Symbol('k', real=True)
        char_eq = D_sym + K * N_sym
        
        st.latex(rf"{sp.latex(char_eq.expand().collect(K))} = 0")
        
        coeffs = sp.Poly(char_eq, s).all_coeffs()
        degree = len(coeffs) - 1
        row0, row1 = coeffs[0::2], coeffs[1::2]
        if len(row0) > len(row1): row1.append(0)
        routh = [row0, row1]
        
        for i in range(2, degree + 1):
            prev1, prev2 = routh[-1], routh[-2]
            new_row = []
            for j in range(len(prev1) - 1):
                if prev1[0] == 0: val = 0 
                else: val = sp.simplify((prev1[0]*prev2[j+1] - prev2[0]*prev1[j+1]) / prev1[0])
                new_row.append(val)
            new_row.append(0)
            routh.append(new_row)
            
        st.markdown("**Tabela de Routh-Hurwitz:**")
        
        def format_val(val):
            if val == 0: return "0"
            val_sym = sp.sympify(val)
            return sp.latex(val_sym)

        import pandas as pd
        table_data = []
        num_cols_max = max(len(r) for r in routh)
        for idx, r_row in enumerate(routh):
            row_dict = {}
            row_dict["Linha"] = f"$s^{{{degree - idx}}}$"
            for c_idx in range(num_cols_max):
                if c_idx < len(r_row):
                    val_str = format_val(r_row[c_idx])
                    if val_str == "0" and c_idx >= len(r_row) - 1:
                        row_dict[f"Coluna {c_idx+1}"] = "0"
                    else:
                        row_dict[f"Coluna {c_idx+1}"] = f"${val_str}$"
                else:
                    row_dict[f"Coluna {c_idx+1}"] = "0"
            table_data.append(row_dict)
            
        df_routh = pd.DataFrame(table_data)
        st.table(df_routh.set_index("Linha"))
        
        valid_ks = []
        if degree >= 2:
            s1_row = routh[-2]
            if K in s1_row[0].free_symbols:
                k_sols = sp.solve(s1_row[0], K)
                valid_ks = [k for k in k_sols if k.is_real and k > 0]
        
        cross_pts = []
        k_crit = None
        if valid_ks:
            aux_row = routh[-3]
            k_crit = valid_ks[0]
            st.markdown(f"Para encontrar a margem de estabilidade, forçamos o primeiro termo da linha $s^{{1}}$ a ser zero:")
            st.latex(rf"{sp.latex(s1_row[0])} = 0")
            
            st.markdown(f"Para o ganho crítico $k = {float(k_crit):.4f}$, a equação auxiliar (da linha $s^{{2}}$) é:")
            aux_eq = sum(coef.subs(K, k_crit) * s**(2*(len(aux_row)-1-j)) for j, coef in enumerate(aux_row) if coef != 0)
            st.latex(rf"{sp.latex(aux_eq.evalf(4))} = 0")
            
            cross_pts = [float(sp.im(r)) for r in sp.roots(aux_eq, s).keys() if sp.re(r) == 0 and sp.im(r) > 0]
            if cross_pts:
                w_cross = cross_pts[0]
                st.markdown(f"**Pontos de cruzamento exatos no eixo imaginário:** $s = -{w_cross:.4f}j, {w_cross:.4f}j$")
        else:
            st.info("🔹 **Aviso:** A tabela de Routh foi calculada, mas não há ganho crítico $k > 0$ com cruzamento no semiplano direito.")

        col_esq, col_centro, col_dir = st.columns([0.5, 5, 0.5])
        with col_centro:
            fig_fin, ax_fin = plt.subplots(figsize=(9, 5))
            
            k_max = float(k_crit) * 3.0 if k_crit is not None else 1000.0
            k_eval = np.linspace(0, k_max, 3000)
            roots_traj = []
            
            den_arr = np.array(den, dtype=float)
            num_arr = np.array(num, dtype=float)
            
            for kv in k_eval:
                poly_k = np.polyadd(den_arr, kv * num_arr)
                roots_traj.append(np.sort_complex(np.roots(poly_k)))
            roots_traj = np.array(roots_traj)
            
            for branch in range(roots_traj.shape[1]):
                ax_fin.plot(np.real(roots_traj[:, branch]), np.imag(roots_traj[:, branch]), 
                            color='#d0d0d0', linewidth=1.5, zorder=2)

            all_p_real = [np.real(p) for p in polos]
            real_roots_sorted = sorted([float(np.real(r)) for r in np.concatenate((polos, zeros)) if np.isreal(r)], reverse=True)
            if real_roots_sorted:
                for i in range(len(real_roots_sorted)):
                    count_right = i + 1
                    if count_right % 2 != 0:
                        start = real_roots_sorted[i]
                        end = real_roots_sorted[i+1] if i+1 < len(real_roots_sorted) else min(all_p_real + [np.real(z) for z in zeros]) - 3.0
                        ax_fin.plot([start, end], [0, 0], color='blue', linewidth=4, label='LGR Eixo Real' if i == 0 else "", zorder=3)

            if np_count > nz_count:
                diff_pz = np_count - nz_count
                sum_p_val = np.sum(np.real(polos))
                sum_z_val = np.sum(np.real(zeros)) if nz_count > 0 else 0.0
                centroide = (sum_p_val - sum_z_val) / diff_pz
                
                x_asym = np.linspace(-10, 5, 400)
                for q in range(diff_pz):
                    ang = ((2*q + 1) * 180) / diff_pz
                    rad = np.deg2rad(ang)
                    if not np.isclose(np.cos(rad), 0, atol=1e-3):
                        y_asym = np.tan(rad) * (x_asym - centroide)
                        if 90 < ang < 270:
                            mask = x_asym <= centroide
                        else:
                            mask = x_asym >= centroide
                        ax_fin.plot(x_asym[mask], y_asym[mask], color='orange', linestyle='--', linewidth=1.5, zorder=4)
                    else:
                        ax_fin.axvline(centroide, color='orange', linestyle='--', linewidth=1.5, zorder=4)

            if 'pontos_validos' in locals() and pontos_validos:
                for pv in pontos_validos:
                    ax_fin.scatter([pv], [0], marker='d', color='green', s=70, zorder=6)

            if cross_pts:
                w = cross_pts[0]
                ax_fin.scatter([0, 0], [w, -w], marker='s', facecolor='cyan', edgecolor='blue', s=80, linewidths=1.5, zorder=7)
                ax_fin.annotate(f"$j\omega = {w:.2f}$", (0, w), textcoords="offset points", xytext=(20, 0),
                                ha='left', va='center', fontsize=9, fontweight='bold', color='blue',
                                bbox=dict(boxstyle="square,pad=0.2", fc="white", ec="blue", lw=1),
                                arrowprops=dict(arrowstyle="->", color="blue", lw=1))
                ax_fin.annotate(f"$j\omega = -{w:.2f}$", (0, -w), textcoords="offset points", xytext=(20, 0),
                                ha='left', va='center', fontsize=9, fontweight='bold', color='blue',
                                bbox=dict(boxstyle="square,pad=0.2", fc="white", ec="blue", lw=1),
                                arrowprops=dict(arrowstyle="->", color="blue", lw=1))

            ax_fin.scatter(all_p_real, [np.imag(p) for p in polos], marker='x', color='red', s=80, label='Polos', linewidths=2, zorder=5)
            if nz_count > 0:
                ax_fin.scatter([np.real(z) for z in zeros], [np.imag(z) for z in zeros], marker='o', 
                               facecolors='none', edgecolors='blue', s=80, label='Zeros', linewidths=2, zorder=5)

            ax_fin.axhline(0, color='black', lw=1)
            ax_fin.axvline(0, color='black', lw=1)
            ax_fin.set_xlim(-5.0, 2.0)
            ax_fin.set_ylim(-10.0, 10.0)
            ax_fin.grid(True, linestyle='--', alpha=0.4)

            from matplotlib.lines import Line2D
            handles, labels = ax_fin.get_legend_handles_labels()
            if np_count > nz_count:
                handles.append(Line2D([0], [0], color='orange', linestyle='--', lw=1.5, label='Assíntotas'))
            if cross_pts:
                handles.append(Line2D([0], [0], marker='s', color='w', markerfacecolor='cyan', markeredgecolor='blue', markersize=8, label='Cruzamento Eixo Imag.'))
            ax_fin.legend(handles=handles, loc='upper right')

            ax_fin.set_title("LGR com Cruzamentos no Eixo Imaginário")
            ax_fin.set_xlabel("Eixo Real ($\sigma$)")
            ax_fin.set_ylabel("Eixo Imaginário ($j\omega$)")
            st.pyplot(fig_fin, use_container_width=True)


        # PASSO 10 ------------------------------------------------
        st.subheader("10. Ângulos de partida/chegada")
        
        complex_poles = [p for p in polos if not np.isclose(np.imag(p), 0, atol=1e-4)]
        
        if len(complex_poles) > 0:
            st.markdown("O ângulo de partida de um pólo complexo conjugado $p$ é dado por:")
            st.latex(r"\theta = 180^\circ - \sum \angle(p - z_i) + \sum \angle(p - p_j)")
            
            angles_dict = {}
            for cp in complex_poles:
                imag_signal = "+" if np.imag(cp) > 0 else "-"
                imag_val = abs(np.imag(cp))
                st.markdown(f"**Para o pólo em $s = {np.real(cp):.1f} {imag_signal} {imag_val:.1f}j$:**")
                
                termos_polos_str = []
                sum_angles_other = 0.0
                
                for p in polos:
                    if np.isclose(p, cp, atol=1e-4): continue
                    diff = cp - p
                    ang_p = np.rad2deg(np.arctan2(np.imag(diff), np.real(diff)))
                    sum_angles_other += ang_p
                    termos_polos_str.append(f"{ang_p:+.1f}^\circ")
                    
                termos_zeros_str = []
                sum_angles_zeros = 0.0
                for z in zeros:
                    diff = cp - z
                    ang_z = np.rad2deg(np.arctan2(np.imag(diff), np.real(diff)))
                    sum_angles_zeros += ang_z
                    termos_zeros_str.append(f"{ang_z:+.1f}^\circ")
                    
                theta = 180.0 - sum_angles_zeros + sum_angles_other
                while theta > 180: theta -= 360
                while theta <= -180: theta += 360
                
                angles_dict[cp] = theta
                detalhe_p = "".join(termos_polos_str)
                detalhe_z = "".join(termos_zeros_str) if termos_zeros_str else "0.0^\circ"
                st.latex(rf"\theta = 180^\circ - ({detalhe_z}) + ({detalhe_p}) = {theta:.1f}^\circ")
        else:
            st.info("🔹 **Aviso:** Este passo não se aplica, pois o sistema não possui pólos ou zeros complexos conjugados (todos são puramente reais). O cálculo de ângulos de partida e chegada é exclusivo para raízes complexas.")

        col_esq, col_centro, col_dir = st.columns([0.5, 5, 0.5])
        with col_centro:
            fig_ang, ax_ang = plt.subplots(figsize=(9, 5))
            
            k_max = 1000.0
            k_eval = np.concatenate([np.linspace(0, 50, 1000), np.logspace(1.7, 4.0, 2000)])
            den_arr = np.array(den, dtype=float)
            num_arr = np.array(num, dtype=float)
            
            prev_roots = np.roots(den_arr)
            branches = [[] for _ in range(len(prev_roots))]
            for kv in k_eval:
                poly_k = np.polyadd(den_arr, kv * num_arr)
                current_roots = np.roots(poly_k)
                matched = np.zeros(len(current_roots), dtype=bool)
                current_sorted = np.zeros_like(current_roots)
                for i, pr in enumerate(prev_roots):
                    distances = [np.abs(pr - cr) if not matched[j] else np.inf for j, cr in enumerate(current_roots)]
                    best_j = np.argmin(distances)
                    matched[best_j] = True
                    current_sorted[i] = current_roots[best_j]
                prev_roots = current_sorted
                for branch_idx, r in enumerate(current_sorted):
                    branches[branch_idx].append(r)
            
            for branch in branches:
                branch_arr = np.array(branch)
                ax_ang.plot(np.real(branch_arr), np.imag(branch_arr), color='#d0d0d0', linewidth=1.5, zorder=2)

            real_roots_sorted = sorted([float(np.real(r)) for r in np.concatenate((polos, zeros)) if np.isreal(r)], reverse=True)
            if real_roots_sorted:
                for i in range(len(real_roots_sorted)):
                    count_right = i + 1
                    if count_right % 2 != 0:
                        start = real_roots_sorted[i]
                        end = real_roots_sorted[i+1] if i+1 < len(real_roots_sorted) else min(all_p_real + [np.real(z) for z in zeros]) - 3.0
                        ax_ang.plot([start, end], [0, 0], color='blue', linewidth=4, label='LGR Eixo Real' if i == 0 else "", zorder=3)

            if np_count > nz_count:
                diff_pz = np_count - nz_count
                sum_p_val = np.sum(np.real(polos))
                sum_z_val = np.sum(np.real(zeros)) if nz_count > 0 else 0.0
                centroide = (sum_p_val - sum_z_val) / diff_pz
                
                x_asym = np.linspace(-10, 5, 400)
                for q in range(diff_pz):
                    ang = ((2*q + 1) * 180) / diff_pz
                    rad = np.deg2rad(ang)
                    if not np.isclose(np.cos(rad), 0, atol=1e-3):
                        y_asym = np.tan(rad) * (x_asym - centroide)
                        if 90 < ang < 270:
                            mask = x_asym <= centroide
                        else:
                            mask = x_asym >= centroide
                        ax_ang.plot(x_asym[mask], y_asym[mask], color='orange', linestyle='--', linewidth=1.5, zorder=4)
                    else:
                        ax_ang.axvline(centroide, color='orange', linestyle='--', linewidth=1.5, zorder=4)

            if len(complex_poles) > 0:
                for cp, th in angles_dict.items():
                    r_cp, i_cp = np.real(cp), np.imag(cp)
                    rad_th = np.deg2rad(th)
                    dx = 1.0 * np.cos(rad_th)
                    dy = 1.0 * np.sin(rad_th)
                    
                    ax_ang.annotate("", xy=(r_cp, i_cp), xytext=(r_cp - dx, i_cp - dy),
                                    arrowprops=dict(arrowstyle="->", color="maroon", lw=2, shrinkA=0, shrinkB=5), zorder=6)
                    
                    text_x = r_cp - 1.8
                    text_y = i_cp + (1.2 if i_cp > 0 else -1.2)
                    ax_ang.text(text_x, text_y, f"{th:.1f}°", fontsize=9, fontweight='bold', color='maroon',
                                ha='center', va='center', bbox=dict(boxstyle="square,pad=0.2", fc="white", ec="maroon", lw=1))

            all_p_real = [np.real(p) for p in polos]
            ax_ang.scatter(all_p_real, [np.imag(p) for p in polos], marker='x', color='red', s=80, label='Polos', linewidths=2, zorder=5)
            if nz_count > 0:
                ax_ang.scatter([np.real(z) for z in zeros], [np.imag(z) for z in zeros], marker='o', 
                               facecolors='none', edgecolors='blue', s=80, label='Zeros', linewidths=2, zorder=5)

            ax_ang.axhline(0, color='black', lw=1)
            ax_ang.axvline(0, color='black', lw=1)
            ax_ang.set_xlim(-8.0, 3.0)
            ax_ang.set_ylim(-8.0, 8.0)
            ax_ang.grid(True, linestyle='--', alpha=0.4)

            from matplotlib.lines import Line2D
            handles, labels = ax_ang.get_legend_handles_labels()
            if np_count > nz_count:
                handles.append(Line2D([0], [0], color='orange', linestyle='--', lw=1.5, label='Assíntotas'))
            ax_ang.legend(handles=handles, loc='upper right')

            ax_ang.set_title("Lugar das Raízes com Vetores de Partida/Chegada")
            ax_ang.set_xlabel("Eixo Real ($\sigma$)")
            ax_ang.set_ylabel("Eixo Imaginário ($j\omega$)")
            st.pyplot(fig_ang, use_container_width=True)

        # PASSO FINAL: GRÁFICO LIMPO DO LGR -----------------------
        st.subheader("🔢 Gráfico Final: Lugar das Raízes")
        
        col_esq, col_centro, col_dir = st.columns([0.5, 5, 0.5])
        with col_centro:
            fig_clean, ax_clean = plt.subplots(figsize=(9, 6))
            
            try:
                import control as ct
                sys_temp = ct.TransferFunction(num, den)
                rlist, klist = ct.root_locus(sys_temp, kvect=np.concatenate([np.linspace(0, 50, 1000), np.logspace(1.7, 5.0, 3000)]), Plot=False)
                for col in range(rlist.shape[1]):
                    ax_clean.plot(np.real(rlist[:, col]), np.imag(rlist[:, col]), linewidth=2, zorder=3)
            except Exception:
                k_eval_clean = np.concatenate([np.linspace(0, 50, 1000), np.logspace(1.7, 5.0, 3000)])
                den_arr = np.array(den, dtype=float)
                num_arr = np.array(num, dtype=float)
                prev_roots = np.roots(den_arr)
                branches = [[] for _ in range(len(prev_roots))]
                for kv in k_eval_clean:
                    poly_k = np.polyadd(den_arr, kv * num_arr)
                    current_roots = np.roots(poly_k)
                    matched = np.zeros(len(current_roots), dtype=bool)
                    current_sorted = np.zeros_like(current_roots)
                    for i, pr in enumerate(prev_roots):
                        distances = [np.abs(pr - cr) if not matched[j] else np.inf for j, cr in enumerate(current_roots)]
                        best_j = np.argmin(distances)
                        matched[best_j] = True
                        current_sorted[i] = current_roots[best_j]
                    prev_roots = current_sorted
                    for branch_idx, r in enumerate(current_sorted):
                        branches[branch_idx].append(r)
                for branch in branches:
                    branch_arr = np.array(branch)
                    ax_clean.plot(np.real(branch_arr), np.imag(branch_arr), linewidth=2, zorder=3)

            all_p_real = [np.real(p) for p in polos]
            all_p_imag = [np.imag(p) for p in polos]
            ax_clean.scatter(all_p_real, all_p_imag, marker='x', color='red', s=100, label='Polos (Malha Aberta)', linewidths=2.5, zorder=5)

            ax_clean.axhline(0, color='black', lw=1)
            ax_clean.axvline(0, color='black', lw=1)
            ax_clean.set_xlim(-8.5, 3.0)
            ax_clean.set_ylim(-7.5, 7.5)
            ax_clean.grid(True, linestyle='--', alpha=0.5)
            
            ax_clean.set_title("Lugar das Raízes")
            ax_clean.set_xlabel("Eixo Real ($\sigma$)")
            ax_clean.set_ylabel("Parte Imaginária")
            ax_clean.legend(loc='upper left')
            
            st.pyplot(fig_clean, use_container_width=True)

        # PASSO 11 e 12 -------------------------------------------
        si = complex(si_real, si_imag)

        st.subheader("11. Critério de Ângulo")
        st.markdown("Dado um ponto $s_i$ no plano $s$, verifica-se se ele pertence ao LGR avaliando se a soma total de fases é um múltiplo ímpar de $180^\circ$:")
        st.latex(r"\left. \angle G(s)H(s) \right|_{s=s_i} = \sum \angle(s_i + z_k) - \sum \angle(s_i + p_j) = \pm 180^\circ (2q + 1)")

        angulos_polos = []
        for p in polos:
            diff = si - p
            ang = np.rad2deg(np.arctan2(np.imag(diff), np.real(diff)))
            angulos_polos.append((p, ang))

        angulos_zeros = []
        for z in zeros:
            diff = si - z
            ang = np.rad2deg(np.arctan2(np.imag(diff), np.real(diff)))
            angulos_zeros.append((z, ang))

        soma_ang_polos = sum(a[1] for a in angulos_polos)
        soma_ang_zeros = sum(a[1] for a in angulos_zeros)
        fase_total = soma_ang_zeros - soma_ang_polos

        fase_mod = (fase_total + 180) % 360 - 180
        pertence = abs(abs(fase_mod) - 180) <= tolerancia

        st.markdown(f"**Avaliação para $s_i = {si_real:+.2f} {si_imag:+.2f}j$:**")
        st.latex(rf"\text{{Fase Total}} = \sum \text{{Zeros}} - \sum \text{{Polos}} = {fase_total:.2f}^\circ")

        if pertence:
            st.success(f"**Resultado:** O ponto $s_i$ **PERTENCE** ao LGR (dentro da tolerância de $\\pm {tolerancia:.1f}^\\circ$).")
        else:
            st.error(f"**Resultado:** O ponto $s_i$ **NÃO PERTENCE** ao LGR (fase de {fase_total:.2f}° fora da tolerância).")

        st.subheader("12. Determinação do Parâmetro $K$ no Ponto $s_i$")
        st.latex(r"K_i = \left. \frac{\prod |s + p_j|}{\prod |s + z_k|} \right|_{s=s_i}")

        prod_dist_polos = np.prod([np.abs(si - p) for p in polos])
        prod_dist_zeros = np.prod([np.abs(si - z) for z in zeros]) if nz_count > 0 else 1.0
        K_i = prod_dist_polos / prod_dist_zeros

        st.latex(rf"K_i = {K_i:.4f}")

        # --- Gráfico Demonstrativo ---
        col_esq, col_centro, col_dir = st.columns([0.5, 5, 0.5])
        with col_centro:
            fig_test, ax_test = plt.subplots(figsize=(9, 5))
            
            try:
                import control as ct
                sys_temp = ct.TransferFunction(num, den)
                rlist, klist = ct.root_locus(sys_temp, kvect=np.concatenate([np.linspace(0, 50, 1000), np.logspace(1.7, 5.0, 3000)]), Plot=False)
                for col in range(rlist.shape[1]):
                    ax_test.plot(np.real(rlist[:, col]), np.imag(rlist[:, col]), linewidth=2, color='#d0d0d0', zorder=2)
            except Exception:
                pass

            real_roots_sorted = sorted([float(np.real(r)) for r in np.concatenate((polos, zeros)) if np.isreal(r)], reverse=True)
            if real_roots_sorted:
                for i in range(len(real_roots_sorted)):
                    count_right = i + 1
                    if count_right % 2 != 0:
                        start = real_roots_sorted[i]
                        end = real_roots_sorted[i+1] if i+1 < len(real_roots_sorted) else min(all_p_real + [np.real(z) for z in zeros]) - 3.0
                        ax_test.plot([start, end], [0, 0], color='blue', linewidth=4, zorder=3)

            ax_test.scatter([si_real], [si_imag], marker='s', color='magenta', s=100, label=f'$s_i$ (K = {K_i:.2f})', zorder=7)
            
            for p, _ in angulos_polos:
                ax_test.plot([np.real(p), si_real], [np.imag(p), si_imag], color='gray', linestyle=':', lw=1.5, zorder=4)
            for z, _ in angulos_zeros:
                ax_test.plot([np.real(z), si_real], [np.imag(z), si_imag], color='blue', linestyle=':', lw=1.5, zorder=4)

            all_p_real = [np.real(p) for p in polos]
            ax_test.scatter(all_p_real, [np.imag(p) for p in polos], marker='x', color='red', s=80, label='Polos', linewidths=2, zorder=5)
            if nz_count > 0:
                ax_test.scatter([np.real(z) for z in zeros], [np.imag(z) for z in zeros], marker='o', 
                               facecolors='none', edgecolors='blue', s=80, label='Zeros', linewidths=2, zorder=5)

            ax_test.axhline(0, color='black', lw=1)
            ax_test.axvline(0, color='black', lw=1)
            ax_test.set_xlim(-8.5, 3.0)
            ax_test.set_ylim(-7.5, 7.5)
            ax_test.grid(True, linestyle='--', alpha=0.4)
            ax_test.legend(loc='upper right')

            ax_test.set_title("Teste de Localização de Raízes ($s_i$)")
            ax_test.set_xlabel("Eixo Real ($\sigma$)")
            ax_test.set_ylabel("Eixo Imaginário ($j\omega$)")
            st.pyplot(fig_test, use_container_width=True)

    except Exception as e:
        st.error(f"Erro: {e}")