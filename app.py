import time


import streamlit as st
from agent.react_agent import ReactAgent

st.title("扫地机器人智能客服")
st.divider() # 分割线

if "agent" not in st.session_state:
    st.session_state["agent"] = ReactAgent()
if"message" not in st.session_state: # 将历史消息添加到里面
    st.session_state["message"] = []
for message in st.session_state["message"]: # 显示历史消息
    st.chat_message(message["role"]).write(message["content"])

prompt = st.chat_input()
if prompt:
    st.chat_message("user").write(prompt)
    st.session_state["message"].append({"role": "user", "content": prompt})

    responses_message = []
    with st.spinner("客服正在思考。。。"):
        res_stream = st.session_state["agent"].execute_stream(prompt) # 回答的内容返回的是迭代器

        def capture(generator, catch_list):
            """
            拿到迭代器里面的内容，并返回
            :param generator:
            :param catch_list:
            :return:
            """
            for chunk in generator: # 得到一段
                catch_list.append(chunk)
                for char in chunk: # 得到段中的字
                    time.sleep(0.02) # 控制打字速度
                    yield char

        st.chat_message("ai").write_stream(capture(res_stream, responses_message))
        st.session_state["message"].append({"role": "ai", "content": responses_message[-1]})
        st.rerun() # 刷新页面


