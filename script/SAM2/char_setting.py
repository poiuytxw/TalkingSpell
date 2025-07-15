
import json
from openai import OpenAI
import os
import base64
import argparse


#  base 64 编码格式
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
def update_json(msg,file_name):
    # 指定要写入的 JSON 文件名
    # 将 messages 写入 JSON 文件
    with open(file_name, 'w', encoding='utf-8') as json_file:
        json.dump(msg, json_file, ensure_ascii=False, indent=4)

    print(f"消息已写入 {file_name}")

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='创建文件夹和子文件夹')
    parser.add_argument('folder_name', type=str, help='主文件夹的名称')
    parser.add_argument('nickname', type=str, help='起一个名字')

    args = parser.parse_args()
    predefined_name=args.folder_name
    nickname=args.nickname
    json_path = f'../../m_settings/messages_{predefined_name}.json'
    char_path = f'../../m_settings/char_{predefined_name}.json'

    # 将xxxx/test.png替换为你本地图像的绝对路径
    base64_image = encode_image(f"D:/GithubDesktopClone/UIST/m_data/raw_img/{predefined_name}/0000.jpg")

    client = OpenAI(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
        api_key=os.getenv('DASHSCOPE_API_KEY'),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    completion = client.chat.completions.create(
        model="qwen-vl-max-latest",
        messages=[
            {
                "role": "system",
                "content": [{"type":"text","text": "You are a helpful assistant."}]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        # 需要注意，传入Base64，图像格式（即image/{format}）需要与支持的图片列表中的Content Type保持一致。"f"是字符串格式化的方法。
                        # PNG图像：  f"data:image/png;base64,{base64_image}"
                        # JPEG图像： f"data:image/jpeg;base64,{base64_image}"
                        # WEBP图像： f"data:image/webp;base64,{base64_image}"
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}, 
                    },
                    {"type": "text", "text": f"请根据图中描绘的这个主体生成一个这个主体的拟人化人设，其中他的名字是{nickname}，请返回一个 JSON string ,并包含以下字段，[name]:人物的名字,[gender]: 性别（男，女，无性别),[age]: 整数，表示年龄,[personality]: 字符串，表示这个人物的说话风格,[story]:字符串，这个人的背景故事,[rvc]: 根据他的人设从（0-中年女，1-青年女，2-儿童女，3-中年男，4-青年男，5-儿童男，6-无性别）中选择一个合适的音色"
                    },
                ],
            }
        ],
    )
    result=completion.choices[0].message.content
    print(completion.choices[0].message.content)
    # 检查是否以反引号开头并删除
    if result.startswith('```'):
        result = result[7:].strip()  # 删除开头的反引号

    # 如果还有结尾的反引号，可以再处理
    if result.endswith('```'):
        result = result[:-3].strip()  # 删除结尾的反引号
    data = json.loads(result)
    print(data['rvc'])
    update_json(data,char_path)