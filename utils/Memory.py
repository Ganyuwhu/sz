import os

"""
    Memory: 用来记录目标需要哪些列表元素，这些元素是作为永久记忆保存下来的
"""


class Memory:
    def __init__(self, file_name):
        self.file_name = file_name  # 目标文件地址

    # 根据target写入数据
    def write(self, target: str, corr_list: list):
        """
        :param target: 目标污染物
        :param corr_list: 模型学习的与目标污染物最相关的几个变量，由列表形式存储
        :return: 将列表写入文件中，文件每一行为target: corr_list[0], corr_list[1], ......
        如果找到了以target对应的那一行，就将这一行替换成corr_list，否则新建一行
        """
        # 将corr_list改为字符串
        corr_str = ", ".join(map(str, corr_list))

        try:
            with open(self.file_name, 'r') as file:
                lines = file.readlines()
        except FileNotFoundError:
            with open(self.file_name, 'w') as file:
                file.write(f"{target}: {corr_str}\n")
            return  # 直接返回，无需后续处理

        found = False
        new_lines = []
        target_prefix = f"{target}: "

        for line in lines:
            if line.startswith(target_prefix):
                # 找到目标行，替换内容
                new_lines.append(f"{target_prefix}{corr_str}\n")
                found = True
            else:
                # 保留其他行不变
                new_lines.append(line)

        # 如果没有找到目标行，追加新行
        if not found:
            new_lines.append(f"{target_prefix}{corr_str}\n")

        # 将更新后的内容写回文件
        with open(self.file_name, 'w') as file:
            file.writelines(new_lines)

    # 根据target读取数据
    def read(self, target):
        try:
            with open(self.file_name, 'r') as file:
                target_prefix = f"{target}: "

                for line in file:
                    line = line.strip()  # 去除首尾空白字符
                    if line.startswith(target_prefix):
                        # 提取corr_list部分
                        corr_str = line[len(target_prefix):]
                        # 将字符串转换回列表
                        corr_list = [item.strip() for item in corr_str.split(',')]
                        # 尝试将元素转换为适当的数据类型
                        processed_list = []
                        for item in corr_list:
                            processed_list.append(item)

                        return processed_list

        except FileNotFoundError:
            pass  # 文件不存在直接返回None


if __name__ == '__main__':
    mem = Memory(r"..\Memories.txt")
    mem.write("NO2", ["wd", "ws", "T", "P", "RH", "PM25", "Bare", "Building", "Forest", "Grass", "OISA", "Road",
                      "Water", "elevation", "slope", "road_density", "building_height", "D2S", "poi_交通设施",
                      "poi_休闲娱乐", "poi_公司企业", "poi_医疗健康", "poi_商务住宅", "poi_旅游景点", "poi_汽车相关",
                      "poi_生活服务", "poi_科教文化", "poi_购物消费", "poi_运动健身", "poi_酒店住宿", "poi_金融机构",
                      "poi_餐饮美食"])
    mem.write("PM25", ["wd", "ws", "T", "P", "RH", "NO2", "Bare", "Building", "Forest", "Grass", "OISA", "Road",
                       "Water", "elevation", "slope", "road_density", "building_height", "D2S", "poi_交通设施",
                       "poi_休闲娱乐", "poi_公司企业", "poi_医疗健康", "poi_商务住宅", "poi_旅游景点", "poi_汽车相关",
                       "poi_生活服务", "poi_科教文化", "poi_购物消费", "poi_运动健身", "poi_酒店住宿", "poi_金融机构",
                       "poi_餐饮美食"])
    mem.write("O3", ["wd", "ws", "T", "P", "RH", "Forest", "Grass", "elevation", "slope", "road_density", "D2S",
                     "poi_公司企业"])
    print(mem.read("NO2"))
    print(mem.read("PM25"))
    print(mem.read("O3"))
