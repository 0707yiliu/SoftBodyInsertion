try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import numpy as np

file_root = "/home/yi/robotic_manipulation/peg_in_hole/ur3_rl_sim2real/gym_envs/models/"

tool_xml = file_root + 'ur5e_gripper/ur5e_with_gripper_softTool.xml'
tree = ET.parse(tool_xml)
root = tree.getroot()
tool_stiffness = np.random.uniform(low=1, high=2)
print(str(tool_stiffness))
i = 0
for student in root.iter('joint'):
    i += 1
    if i == 2:
        student.set("stiffness", str(tool_stiffness))
        # student.attrib["stiffness"] = 0.08
        print(student.attrib) # 打印名字

tree.write(file_root+'ur5e_gripper/ur5e_with_gripper_softTool_test.xml')

        
