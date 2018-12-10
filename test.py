from operator import itemgetter



if __name__ == '__main__':
    treedata = [[1, 2, 3, 4, 5],[9, 5, 1],[0]]
    labeldata = [[11, 12, 13, 13, 14],[3, 4, 5],[0]]
    for i in range(len(treedata)):
        new_treedata = []
        new_labeldata = []
        stock_len = len(treedata[i])
        treedata_set = []
        for j in range(stock_len):
            treedata_set.append((j, treedata[i][j]))
        treedata_reset = sorted(treedata_set, key=itemgetter(1))
        reserve_set = []
        for j in range((int)(stock_len*0.3), (int)(stock_len*0.7)):
            reserve_set.append(treedata_reset[j][0])
        for j in range(stock_len):
            if j not in reserve_set:
                new_treedata.append(treedata[i][j])
                new_labeldata.append(labeldata[i][j])
        print(new_treedata)
        print(new_labeldata)