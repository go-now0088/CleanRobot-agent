from utils.path_tool import get_abs_path
from utils.config_handler import prompts_config
from utils.logger_handler import  logger

def load_system_prompts():
    try:
        system_prompts_path = get_abs_path(prompts_config["mian_prompt_path"])
    except KeyError as e:
        logger.error(f"提示词模版不存在yml中")
        raise e
    try:
        return open(system_prompts_path, "r", encoding="utf-8").read()
    except Exception  as e:
        logger.error(f"解析system提示词模版出错， {str(e)}")
        raise e

def load_rag_prompts():
    try:
        rag_prompts_path = get_abs_path(prompts_config["rag_summarize_prompt_path"])
    except KeyError as e:
        logger.error(f"提示词模版不存在yml中")
        raise e
    try:
        return open(rag_prompts_path, "r", encoding="utf-8").read()
    except Exception  as e:
        logger.error(f"解析rag提示词模版出错， {str(e)}")
        raise e

def load_report_prompts():
    try:
        report_prompts_path = get_abs_path(prompts_config["report_prompt_path"])
    except KeyError as e:
        logger.error(f"提示词模版不存在yml中")
        raise e
    try:
        return open(report_prompts_path, "r", encoding="utf-8").read()
    except Exception  as e:
        logger.error(f"解析report提示词模版出错， {str(e)}")
        raise e

if __name__ == '__main__':
    print(load_system_prompts())