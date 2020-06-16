from typing import List
import pandas as pd
import numpy as np



class CheckNode:
    def __init__(self, lst):
        self.id, self.x, self.y, self.z, t1, t2 = lst
        self.loc = np.array([self.x, self.y, self.z], dtype=np.double)
        self.check_type = None
        if t1 == 1:
            self.check_type = "V"
        elif t1 == 0:
            self.check_type = "H"
        else:
            raise RuntimeError
        if t2 == 1:
            self.is_success = False
        elif t2 == 0:
            self.is_success = True




def load_data(s="附件2：数据集2-终稿.xlsx"):

    xlsx = pd.read_excel(s)

    # print(xlxs.iloc[1].to_numpy())

    # print(xlxs[2:-1].to_numpy())
    check_node_lst = []  # type:List[CheckNode]
    for i in xlsx[2:-1].to_numpy():
        check_node_lst.append(CheckNode(i))

    # print(len(check_node_lst))
    # print()
    START = np.array(xlsx.iloc[1].to_numpy()[1:4], dtype=np.double)
    END = np.array(xlsx.iloc[-1].to_numpy()[1:4], dtype=np.double)
    # print(START)
    # print(END)
    return check_node_lst, START, END
