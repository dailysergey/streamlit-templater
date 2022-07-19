import pickle
from typing import List
from datasets import Dataset
import pandas as pd
import numpy as np
import json
from json import JSONEncoder
from datasets import load_dataset, load_from_disk
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import streamlit as st
from functools import wraps
import time
from ast import literal_eval
from collections import Counter
from operator import itemgetter
import os
import urllib

# download with progress bar
mybar = None
main_dir = "https://github.com/dailysergey/streamlit-templater/releases/download/files"

def show_progress(block_num, block_size, total_size):
    global mybar
    if mybar is None:
        mybar = st.progress(0.0)
    downloaded = block_num * block_size / total_size
    if downloaded <= 1.0:
        mybar.progress(downloaded)
    else:
        mybar.progress(1.0)


necessary_files = ['products_name.csv',
                   'okpd2.csv', 'ktru.csv']
for file_name in necessary_files:

    # download files locally
    if not os.path.isfile(file_name):
        with st.spinner('Скачиваем файлы. Это делается один раз и занимает минуту...'):
            try:
                #st.info(f'{file_name} скачивается')
                print(f'{file_name} скачивается')
                urllib.request.urlretrieve(main_dir, file_name, show_progress)
                #st.success(f'{file_name} скачался')
                print((f'{file_name} скачался'))
            except Exception as e:
                #st.error(f'{file_name} не скачался. Ошибка: {e}')
                print(f'{file_name} не скачался. Ошибка: {e}')

def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(
            f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper


# Page layout
# Page expands to full width
st.set_page_config(page_title="ГИСП подбор шаблонов", page_icon="🗂", layout="wide")
st.markdown('<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-1BmE4kWBq78iYhFldvKuhfTAU6auU8tT94WrHftjDbrCEXSU1oBoqyl2QvZ6jIW3" crossorigin="anonymous">', unsafe_allow_html=True)


@st.experimental_singleton
def initialize():
    device = torch.device('cpu')
    print('Device:', device)

    products = pd.read_csv(f'{main_dir}/prod_only_names.csv')
    # prod_hf_ds = load_from_disk('products_embeddings.hf')
    #prod_hf_ds = load_from_disk('products_name_embeddings.hf')
    prod_hf_ds = load_dataset('gusevski/products_embeddings')
    prod_hf_ds.add_faiss_index(column="embedding")

    model_ckpt = "cointegrated/rubert-tiny2"
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    model = AutoModel.from_pretrained(model_ckpt)
    return model, tokenizer, device, prod_hf_ds, products


@st.cache(allow_output_mutation=True)
def get_products_df(path):

    products_df = pd.read_csv(path)
    products_df['okpd2'].fillna(0, inplace=True)
    products_df['ktru'].fillna(0, inplace=True)
    return products_df


@st.cache(allow_output_mutation=True)
def get_okpd2_ktru():
    okpd2_df = pd.read_csv(f"{main_dir}/okpd2.csv", index_col="id")
    okpd2_df.loc[0] = {'name': "Не определено"}
    ktru_df = pd.read_csv(f"{main_dir}/ktru.csv", index_col="id")
    ktru_df.loc[0] = {'name': "Не определено"}
    return okpd2_df, ktru_df


def get_group_and_cluster_number(query: str):
    query_embedding = embed_bert_cls([query.lower()])
    scores, samples = prod_hf_ds.get_nearest_examples(
        "embedding", query_embedding, k=15)

    samples_df = pd.DataFrame.from_dict(samples)
    samples_df["scores"] = scores
    samples_df.sort_values("scores", ascending=True, inplace=True)
    #group_number, cluster_number = vote_for_cluster(samples_df.id.values)
    group_numbers = vote_for_okpd(samples_df.id.values)
    # return group_number, cluster_number
    return group_numbers


@timeit
def make_template(df: pd.DataFrame, group_number: int, cluster_num: int):

    group = df[df.group_number == group_number]

    result = []
    template = dict()
    values = dict()
    group_size = group[group.cluster_number == cluster_num].shape[0]
    for field in group[group.cluster_number == cluster_num]['fields']:

        cur_field = literal_eval(field)

        for key in cur_field.keys():

            if key in template:
                template[key] += 1
                values[key].add(cur_field[key])
            else:
                template[key] = 1
                values[key] = set([cur_field[key]])

    result.append({cluster_num: template})
    return result, values, group_size


df = get_products_df('products_df_rubert-tiny2_full_descr.csv')
okpd2_df, ktru_df = get_okpd2_ktru()
model, tokenizer, device, prod_hf_ds, products = initialize()


def card(id_val, source, context):
    st.markdown(f"""
    <div class="card" style="margin:1rem; background-color:lightgrey;">
        <div class="card-body">
            <h5 class="card-title">{source}</h5>
            <h6 class="card-subtitle mb-2 text-muted">{id_val}</h6>
            <p class="card-text">{context}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)


def color_survived(val):
    color = 'green' if val else 'red'
    return f'background-color: {color}'


def vote_for_cluster(ids: List[str]):
    try:
        group_nums = df[df.id.isin(ids)].group_number.values
        print('1', group_nums)

        group_num = ((list(Counter(group_nums).most_common(1))))[0][0]
        print('2', group_num)
        cluster_nums = df[df.group_number == group_num].cluster_number.values
        print('3', cluster_nums)

        cluster_num = ((list(Counter(cluster_nums).most_common(1))))[0][0]
        print('4', cluster_num)
        return group_num, cluster_num
    except Exception as e:
        print(e)
        return -1, -1


def vote_for_okpd(ids: List[str]):
    try:
        group_nums = df[df.id.isin(ids)].group_number.values
        return group_nums
    except Exception as e:
        print(e)
        return -1


@timeit
def embed_bert_cls(text):
    model.to(device)
    # use padding, truncation of long sequences and return pytorch tensors
    t = tokenizer(text, padding=True, truncation=True,
                  max_length=128, return_tensors='pt')
    t = {k: v.to(model.device) for k, v in t.items()}

    with torch.no_grad():
        # move all tensors on the same device as model
        model_output = model(**t)
    # use only first [CLS] token vector
    embeddings = model_output.last_hidden_state[:, 0, :]
    # normalize vector for easier convergence
    embeddings = torch.nn.functional.normalize(embeddings)
    return embeddings[0].cpu().numpy()  # return result as numpy vector


# CSS to inject contained in a string
hide_table_row_index = """
            <style>
            tbody th {display:none}
            .blank {display:none}
            </style>
            """
# Inject CSS with Markdown
st.markdown(hide_table_row_index, unsafe_allow_html=True)

with open('ktru.pickle', 'rb') as handle:
    ktry = pickle.load(handle)
with open('okpd.pickle', 'rb') as handle:
    okpd = pickle.load(handle)

st.write("""
# Давайте подберем вам шаблон!
""")
query = st.text_input('Введите название товара', '',
                      max_chars=50, help='только название товара')
character = pd.DataFrame()
if query != "":

    #group_number, cluster_number = get_group_and_cluster_number(query)
    # if group_number == -1 and cluster_number == -1:
    #    group_number, cluster_number = get_group_and_cluster_number(query.split(' ')[0])
    group_numbers = get_group_and_cluster_number(query)

    if group_numbers[0] == -1:
        st.write('Запутался, не просите повторить...')
        query = ''
    else:
        okpds_unique = df[df.group_number.isin(group_numbers)].okpd2.unique()
        okpd2_template = st.selectbox('Список подходящих ОКПД2',
                                      okpd2_df[okpd2_df.index.isin(
                                          okpds_unique)].index,
                                      format_func=lambda x: okpd2_df.loc[int(x)][['code', 'name']].values)
        print('okpd2_template', okpd2_template)
        template = df[df.okpd2 == okpd2_template]

        group_number = template.group_number.values[0]
        cluster_number = template.cluster_number.mode().values[0]
        print('Group number', group_number, 'cluster_number', cluster_number)

        template_result, values, group_size = make_template(
            df, group_number, cluster_number)

        okpd2_id = df[(df.group_number == group_number) & (
            df.cluster_number == cluster_number)].okpd2.values[0]
        ktru_id = df[(df.group_number == group_number) & (
            df.cluster_number == cluster_number)].ktru.values[0]
        okpd_template = okpd2_df[okpd2_df.index ==
                                 int(okpd2_id)].code.values[0]
        ktru_template = ktru_df[ktru_df.index == int(ktru_id)].code.values[0]

        st.write(
            f"ОКПД2: { okpd2_df[okpd2_df.index == int(okpd2_id)][['code','name']].values }")
        st.write(
            f"КТРУ: { ktru_df[ktru_df.index == int(ktru_id)][['code','name']].values }")

        # if okpd_template in okpd:
        #    st.write(okpd[okpd_template])
        # if ktru_template in ktry:
        #    st.write(ktry[ktru_template])

        cur_template = sorted(list(template_result[0].values())[
            0].items(), key=itemgetter(1), reverse=True)

        character = pd.DataFrame(cur_template)

if len(character) != 0:
    character.columns = ['Тип характеристики', 'Частота встречаемости']
    character['Возможные значения'] = character['Тип характеристики'].map(
        values)
    character['Частота встречаемости, %'] = character['Частота встречаемости'].apply(
        lambda x: (x/group_size) * 100)
    min_val = st.number_input("Выберите минимальную частоту встречаемости, %",
                              min_value=0,
                              value=int(
                                  np.min(character['Частота встречаемости, %'].values)),
                              max_value=100)

    st.write("{} Записей ".format(
        str(character[character['Частота встречаемости, %'] >= min_val].shape[0])))

    def check_characteristic(val):

        color = 'transparent'
        # if val in ktry[ktru_template].characteristics:
        # print(okpd[okpd_template]['characteristics'])
        if val in ktry[ktru_template]['characteristics']:
            color = 'pink'
        return f'background-color: {color}'
    print(ktru_template)
    # if ktru_template in ktry:
    if okpd_template in okpd:
        st.dataframe(character[character['Частота встречаемости, %'] >= min_val][['Тип характеристики', 'Возможные значения',
                     'Частота встречаемости, %']].style.applymap(check_characteristic, subset=['Тип характеристики']))
    else:
        st.dataframe(character[character['Частота встречаемости, %'] >= min_val][[
                     'Тип характеристики', 'Возможные значения', 'Частота встречаемости, %']])


# '''if group_number == -1 and cluster_number == -1:
#        st.write('Запутался, не просите повторить...')
#        query = ''
#    else:
#       template_result, values, group_size = make_template(
#            df, group_number, cluster_number)
#
#
#        okpd2_id = df[(df.group_number == group_number) & (
#            df.cluster_number == cluster_number)].okpd2.values[0]
#        ktru_id = df[(df.group_number == group_number) & (
#            df.cluster_number == cluster_number)].ktru.values[0]
#       '''
