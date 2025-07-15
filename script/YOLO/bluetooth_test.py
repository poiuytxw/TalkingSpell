import asyncio
from bleak import BleakClient
import keyboard

SERVICE_UUID = "12345678-1234-5678-1234-56789abcdef0"
CHARACTERISTIC_UUID = "12345678-1234-5678-1234-56789abcdef1"
CHARACTERISTIC_UUID_2="12345678-1234-5678-1234-56789abcdef2"

async def connect_and_receive(address):
    async with BleakClient(address) as client:
        print("Connected to ESP32")
        while True:
            value = await client.read_gatt_char(CHARACTERISTIC_UUID)
            print("Received:", value.decode())
            UpdateJson(key='state', value=str(value.decode()))
            
             # 检测 's' 按键
            if keyboard.is_pressed('s'):  
                await send_signal(client)
                await asyncio.sleep(0.5)  # 防止重复发送

async def send_signal(client):
    message = "button pressed"
    await client.write_gatt_char(CHARACTERISTIC_UUID_2, message.encode())
    print("Signal sent to ESP32: ", message)
import json

def UpdateJson(file_path='recording_state.json', key=None, value=None):
    try:
        # 读取现有数据
        with open(file_path, 'r') as file:
            data = json.load(file)

        # 更新特定键的值
        if key is not None:
            data[key] = value

        # 写入更新后的数据
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)

        print("Data updated successfully.")

    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
    except json.JSONDecodeError:
        print("Error: Failed to decode JSON.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    address = "d8:3b:da:89:4a:f9"  # 或使用 "ESP32_S3_BLE"
    asyncio.run(connect_and_receive(address))