"""
为整个工程提供一个绝对路径
"""
import os

def get_root_path():
    current_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # 返回到当前位置的上两级
    return current_path

def get_abs_path(relative_path):
    return os.path.abspath(os.path.join(get_root_path(), relative_path)) # 将相对路径和根路径拼接起来得到绝对路径

if __name__ == '__main__':
    print(get_abs_path("config/config.json"))
