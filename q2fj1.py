from copy import deepcopy

from config1 import *
from public_function import *

PATH = []
best_PATH = []
best_length = float("inf")


def go(loc, dh, dv, flight_length, pre_loc):
    global best_length
    # print(len(PATH))
    if flight_length > best_length:
        return
    PATH.append(loc)
    if flight_length != 0 and dh + cal_add_delta2(pre_loc, loc, END) < theta and dv + cal_add_delta2(pre_loc, loc,
                                                                                                     END) < theta:
        # print("终点id",dv + cal_add_delta(loc, END),  dh + cal_add_delta(loc, END),  "终点")
        if flight_length + cal_length2(pre_loc, loc, END) < best_length:
            best_PATH = deepcopy(PATH)
            best_length = flight_length + cal_length2(pre_loc, loc, END)
            print(best_PATH)
            print(best_length)
            print(len(best_PATH))
        PATH.pop()
        return "OK"
    ok_node_list = []
    for i in check_node_lst:
        # print(path)
        for j in PATH:
            if i.loc is j:
                # print(i.loc, j)
                break
            # else:
            #     print(PATH)
        else:
            add_delta = cal_add_delta2(pre_loc, loc, i.loc)
            if i.check_type == "V":
                if dv + add_delta <= alpha1 and dh + add_delta <= alpha2:
                    ok_node_list.append((cal_length(i.loc, END), i))

            if i.check_type == "H":
                if dv + add_delta <= beta1 and dh + add_delta <= beta2:
                    ok_node_list.append((cal_length(i.loc, END), i))
    ok_node_list.sort(key=lambda x: x[0])

    # print([i[1].loc for i in ok_node_list])

    for i in ok_node_list[:4]:
        if flight_length != 0:

            add_delta = cal_add_delta2(pre_loc, loc, i[1].loc)
            add_length = add_delta * 1000
        else:
            add_delta = cal_add_delta(loc, i[1].loc)
            add_length = add_delta * 1000
        if i[1].check_type == "V":
            # print(i[1].id,  dv + add_delta,dh + add_delta, "垂直")
            go(i[1].loc, dh + add_delta, 0, flight_length + add_length, loc)

        if i[1].check_type == "H":
            # print(i[1].id,  dv + add_delta,dh + add_delta, "水平")
            go(i[1].loc, 0, dv + add_delta, flight_length + add_length, loc)
    PATH.pop()


if __name__ == "__main__":
    go(START, 0, 0, 0, np.empty(3, dtype=np.double))
