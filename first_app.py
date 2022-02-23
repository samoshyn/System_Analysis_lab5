import streamlit as st
import seaborn as sns
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
import itertools

st.set_page_config(layout="wide")

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

hide_footer_style = """
    <style>
    .reportview-container .main footer {visibility: hidden;}    
    """
st.markdown(hide_footer_style, unsafe_allow_html=True)

def get_T_agg_names_df(df):
    arr_s = []
    arr_w = []
    s_df, w_df = df.loc[s_inds], df.loc[w_inds]
    for col in t_cols:
        mask_s = s_df[col]>0
        mask_w = w_df[col]>0
        arr_s.append(mask_s[mask_s].index.tolist())
        arr_w.append(mask_w[mask_w].index.tolist())

    df_t_name = pd.concat([pd.Series(arr_s, name='Strengths that can help implementation'),
                      pd.Series(arr_w, name='Weaknesses that may interfere with implementation')], axis=1)
    df_t_name.index = t_cols
    return df_t_name

def get_T_agg_sum_df(df):
    arr_s = []
    arr_w = []
    s_df, w_df = df.loc[s_inds], df.loc[w_inds]
    for col in t_cols:
        arr_s.append(s_df[col].sum())
        arr_w.append(w_df[col].sum())

    df_t = pd.concat([pd.Series(arr_s, name='Influence of strong characteristics'),
                          pd.Series(arr_w, name='Influence of weak characteristics')], axis=1)
    df_t['H'] = df_t['Influence of strong characteristics'] - df_t['Influence of weak characteristics']
    df_t.index = t_cols
    return df_t

def get_O_agg_names_df(df):
    arr_s = []
    arr_w = []
    s_df, w_df = df.loc[s_inds], df.loc[w_inds]
    for col in o_cols:
        mask_s = s_df[col]>0
        mask_w = w_df[col]>0
        arr_s.append(mask_s[mask_s].index.tolist())
        arr_w.append(mask_w[mask_w].index.tolist())

    df_o_name = pd.concat([pd.Series(arr_s, name='Strengths that can offset the threat'),
                      pd.Series(arr_w, name='Weaknesses that can increase the threat')], axis=1)
    df_o_name.index = o_cols
    return df_o_name

def get_O_agg_sum_df(df):
    arr_s = []
    arr_w = []
    s_df, w_df = df.loc[s_inds], df.loc[w_inds]
    for col in o_cols:
        arr_s.append(s_df[col].sum())
        arr_w.append(w_df[col].sum())

    df_o = pd.concat([pd.Series(arr_s, name='Influence of strong characteristics'),
                          pd.Series(arr_w, name='Influence of weak characteristics')], axis=1)
    df_o['D'] = df_o['Influence of strong characteristics'] - df_o['Influence of weak characteristics']
    df_o.index = o_cols
    return df_o

def get_S_agg_sum_df(df):
    arr_t = []
    arr_o = []
    t_df, o_df = df[t_cols], df[o_cols]
    for ind in s_inds:
        arr_t.append(t_df.loc[ind].sum())
        arr_o.append(o_df.loc[ind].sum())

    df_s = pd.concat([pd.Series(arr_o, name='Opportunities that can enhance'),
                              pd.Series(arr_t, name='Threats that could weaken')], axis=1)
    df_s['F'] = df_s['Opportunities that can enhance'] + df_s['Threats that could weaken']
    df_s.index = s_inds
    return df_s

def get_W_agg_sum_df(df):
    arr_t = []
    arr_o = []
    t_df, o_df = df[t_cols], df[o_cols]
    for ind in w_inds:
        arr_t.append(t_df.loc[ind].sum())
        arr_o.append(o_df.loc[ind].sum())

    df_w = pd.concat([pd.Series(arr_o, name='Opportunities that can enhance'),
                              pd.Series(arr_t, name='Threats that could weaken')], axis=1)
    df_w['G'] = df_w['Opportunities that can enhance'] + df_w['Threats that could weaken']
    df_w.index = w_inds
    return df_w

def change_colour(val):
    return ['background-color: #e6e6e6' if x in target_num else 'background-color: #a3a3a3' for x in val]

def get_design_table(df, ascending=False):
    
    columns = df.columns.tolist()
    if not ascending:
        subset = df[df[columns[1]]>=0]
    else:
        subset = df[df[columns[1]]<=0]
    
    global target_num
    target_num = set(subset.head(int(len(subset)*0.5) + 1)[columns[0]])
    
    return df.style.apply(change_colour, axis=1, subset=[df.columns[0]]).bar(subset=[columns[1]], align='mid', color=['#d65f5f', '#5fba7d']), target_num

def get_swot_analysis(df, dict_characteristics):
    
    cols = df.columns
    inds = df.index
    
    global t_cols, o_cols, s_inds, w_inds
    t_cols = [col for col in cols if 'T' in col]
    o_cols = [col for col in cols if 'O' in col]
    s_inds = [ind for ind in inds if 'S' in ind]
    w_inds = [ind for ind in inds if 'W' in ind]
    
    df_t_name = get_T_agg_names_df(df)
    df_t_sum = get_T_agg_sum_df(df)['H'].reset_index().rename(columns={'index':'Threat num'}).sort_values(by=['H'])
    
    df_o_name = get_O_agg_names_df(df)
    df_o_sum = get_O_agg_sum_df(df)['D'].reset_index().rename(columns={'index':'Opportunity num'}).sort_values(by=['D'], ascending=False)

    df_s_sum = get_S_agg_sum_df(df)['F'].reset_index().rename(columns={'index':'Strengths num'}).sort_values(by=['F'], ascending=False)
    df_w_sum = get_W_agg_sum_df(df)['G'].reset_index().rename(columns={'index':'Weakness num'}).sort_values(by=['G'], ascending=False)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        d, target_s = get_design_table(df_s_sum, ascending=False)
        st.table(d)
    with col2:
        d, target_w = get_design_table(df_w_sum, ascending=False)
        st.table(d)
    with col3:
        d, target_o = get_design_table(df_o_sum, ascending=False)
        st.table(d)
    with col4:
        d, target_t = get_design_table(df_t_sum, ascending=True)
        st.table(d)
    with col5:
        st.write(dict_characteristics)
        
    return {"S": target_s, "W": target_w, "O": target_o, "T": target_t}
        
def get_topsis_analysis(strategies, dict_characteristics):
    
    cols = ['SO','WO','ST','WT']
    
    wi, data = strategies[['Wi']], strategies.drop(columns=['Wi'])
    wi['Wi'] /= wi['Wi'].sum()
    
    st.caption('1. Вхідні дані:')
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(data)
    with col2:
        st.write(wi)
    with col3:
        st.write(dict_characteristics)
    
    for col in data.columns:
        sum_norm = np.sqrt(np.sum(data[col]**2))
        data[col] /= sum_norm
    with col1:
        st.caption('2. Нормалізація:')
        st.write(data)
        
    data = np.round(data,5)
    for col in data.columns:
        data[col]*=wi['Wi']
    
    a_plus = np.max(data, axis=1)
    a_min = np.min(data, axis=1)
    data['A+'] = a_plus
    data['A-'] = a_min
    with col2:
        st.caption("3. Зважена матриця, найкращі та найгірші розв'язки")
        st.write(data)
    
    strategy = pd.DataFrame(columns=cols)
    strategy = strategy.append(pd.DataFrame(
        [[np.sqrt(np.sum((data[col]-data['A+'])**2)) for col in cols]],
        index=['Dj+'],
        columns=cols
    ))
    strategy = strategy.append(pd.DataFrame(
        [[np.sqrt(np.sum((data[col]-data['A-'])**2)) for col in cols]],
        index=['Dj-'],
        columns=cols
    ))
    strategy = strategy.append(
        (strategy.loc['Dj-']/(strategy.loc['Dj-']+strategy.loc['Dj+'])).to_frame(name='Cj*+').T
    )
    with col1:
        st.caption('4. Коефіцієнти близькості кожної альтернативи:')
        st.write(strategy)
    
    with col2:
        st.caption('5. Результати методу:')
        st.write(strategy.T[['Cj*+']].sort_values(by='Cj*+', ascending=False).style.background_gradient(cmap=sns.light_palette("green", as_cmap=True)))
        
        
def get_vikor_analysis(strategies, dict_characteristics):
    
    cols = ['SO','WO','ST','WT']
    
    wi, data = strategies[['Wi']], strategies.drop(columns=['Wi'])
    wi['Wi'] /= wi['Wi'].sum()
    data['f*'] = data[cols].max(1)
    data['f**'] = data[cols].min(1)
    
    st.caption('1. Вхідні дані:')
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(data)
    with col2:
        st.write(wi)
    with col3:
        st.write(dict_characteristics)
    
    Si = [np.sum(wi['Wi']*(data['f*']-data[col])/(data['f*']-data['f**'])) for col in cols]
    Rj = [np.max(wi['Wi']*(data['f*']-data[col])/(data['f*']-data['f**'])) for col in cols]
    Q_arr = []
    v = 0.005
    

        
    Si_row = Si + [np.max(Si)] + [np.min(Si)]
    Rj_row = Rj + [np.max(Rj)] + [np.min(Rj)]
    add_df = pd.DataFrame(
        np.vstack([Si_row,Rj_row,]),
        index=['S','R'],
        columns=data.columns
    )
    data = data.append(add_df)
    
    for col in cols:
        sum_1 = v*(data.loc['S',col]-data.loc['S','f**'])/(data.loc['S','f*']-data.loc['S','f**'])
        sum_2 = (1-v)*(data.loc['R',col]-data.loc['R','f**'])/(data.loc['R','f*']-data.loc['R','f**'])
        Qj = sum_1+sum_2
        Q_arr.append(Qj)
        
    data = data.append(
        pd.DataFrame([Q_arr+[np.max(Q_arr)]+[np.min(Q_arr)]],
                     index=['Q'],
                     columns=data.columns)
    )
    
    
    with col1:
        st.caption('2. Обчислені S, R, Q:')
        st.write(data)
        
    key_ind = ['S','R','Q']

    res = pd.DataFrame(
        [
            data.loc[ind,cols].sort_values(ascending=False).index.tolist()[::-1]
            for ind in key_ind
        ], index=key_ind
    )
    
    def f(dat, c='green'):
        return [f'background-color: {c}' for i in dat]
    with col2:
        st.caption("3. Результати методу:")
        st.write(res.style.apply(f, axis=0, subset=[0]))
    
    
def get_df(file):
    
  extension = file.name.split('.')[1]
  if extension.upper() == 'CSV':
      df = pd.read_csv(file)
  elif extension.upper() == 'XLSX':
      df = pd.read_excel(file, engine='openpyxl')

  return df


def main():
    
    df = None
    dict_characteristics = None
    strategies = None
    st.header('Налаштування')
    st.info("Завантажте матрицю зіставлення компонентів SWOT-аналізу")
    uploaded_file = st.file_uploader("Завантажити матрицю")
    if uploaded_file is not None:
        df = get_df(uploaded_file).set_index('Unnamed: 0')
        
        st.info("Завантажте розшифрування характеристик системи (опціонально)")
        uploaded_file = st.file_uploader("Завантажити характеристики")
        if uploaded_file is not None:
            dict_characteristics = get_df(uploaded_file)
            
        st.info("Завантажте стратегії та ваги для аналізу")
        uploaded_file = st.file_uploader("Завантажити стратегії та ваги")
        if uploaded_file is not None:
            strategies = get_df(uploaded_file).set_index('Unnamed: 0')
        
            go = st.button('Виконати')
            if go:
                st.subheader('Результати SWOT-аналізу')
                select_characts = get_swot_analysis(df, dict_characteristics)
                st.subheader('Метод TOPSIS')
                get_topsis_analysis(strategies.loc[[i for x in select_characts.values() for i in x]], dict_characteristics)
                
                st.subheader('Метод VIKOR')
                get_vikor_analysis(strategies.loc[[i for x in select_characts.values() for i in x]], dict_characteristics)
            
main()
