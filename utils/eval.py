
class TreeStructureEval():
    def __init__(self, pred_data, gold_data):
        self.pred_data = pred_data
        self.gold_data = gold_data

    # 将符合诊疗决策树约束的节点前序序列转化为代表诊疗决策树结构的节点矩阵，matrix[i][j]='F'/'L'/'R'表示第j个节点是第i个节点的父/左子/右子节点
    def nodematrix(self, tree):
        nodelist = []
        for i in range(len(tree)):
            nodelist.append(tree[i]["role"])
        # print("nodelist", nodelist)
        node_matrix = [[0 for i in range(len(nodelist))] for j in range(len(nodelist))]
        while (nodelist[0] != 'D'):
            for i in range(len(nodelist)):
                flag, leaf1, leaf2 = 0, 0, 0
                for j in range(i + 1, len(nodelist)):
                    if nodelist[j] == 'D' and flag == 0:
                        flag = 1
                        leaf1 = j
                    elif nodelist[j] == 'X':
                        continue
                    elif nodelist[j] == 'D' and flag == 1:
                        # print(i)
                        leaf2 = j
                        nodelist[i] = 'D'
                        node_matrix[leaf1][i] = 'F'
                        node_matrix[leaf2][i] = 'F'
                        node_matrix[i][leaf1] = 'L'
                        node_matrix[i][leaf2] = 'R'
                        for k in range(i + 1, leaf2 + 1):
                            nodelist[k] = 'X'
                        # print((nodelist))
                        break
                    elif nodelist[j] == 'C':
                        break

        return (node_matrix)

    # 计算两个节点的距离
    def node_dis(self, node1, node2):
        if node2 is None:
            node2 = {"role": "", "triples": [], "logical_rel": ""}
        dis = 0
        if node1["role"] != node2["role"]:
            dis += 1
        # print(dis)
        if node1["logical_rel"] != node2["logical_rel"]:
            dis += 1
        list1, list2 = [], []
        for triple in node1["triples"]:
            list1.append((triple[0].lower(), triple[1].lower(), triple[2].lower()))
        for triple in node2["triples"]:
            list2.append((triple[0].lower(), triple[1].lower(), triple[2].lower()))
        dis += len(list((set(list1) | set(list2)) - (set(list1) & set(list2))))
        return (dis)
    
    def is_tree_equal(predict_tree,gold_tree):
        if len(predict_tree) != len(gold_tree):
            return 0
        else:
            for i in range(len(predict_tree)):
                if predict_tree[i]['role'] == gold_tree[i]['role'] and predict_tree[i]['logical_rel'] == predict_tree[i]['logical_rel'] and set(
                    predict_tree[i]['triples']) == set(gold_tree[i]['triples']):
                    continue
                else:
                    return 0
        return 1

    def is_path_equal(self,path1, path2):
        if (len(path1) != len(path2)):
            return False
        for i in range(len(path1)):
            if isinstance(path1[i], dict) and isinstance(path2[i], dict):
                if path1[i]['role'] == path2[i]['role'] and path1[i]['logical_rel'] == path2[i]['logical_rel'] and set(
                        path1[i]['triples']) == set(path2[i]['triples']):
                    continue
                else:
                    return False
            elif path1[i] != path2[i]:
                return False
        return True

    # 计算模型预测的诊疗决策树和ground turth的距离，距离越小表示两树越相似
    def edit_distance(self, predict_tree, gold_tree, predict_matrix, gold_matrix):
        dis = 0
        stack1 = [0]
        stack2 = [0]
        while stack1:
            s1 = stack1.pop()
            s2 = stack2.pop()
            if ('L' not in predict_matrix[s1] and 'R' not in predict_matrix[s1]) and (
                    'L' in gold_matrix[s2] or 'R' in gold_matrix[s2]):
                # print("计算树1节点" + str(s1) + "计算树2节点" + str(s2)+"和它的子节点")
                dis += self.node_dis(predict_tree[s1], gold_tree[s2])
                stack_tmp = []
                stack_tmp.append(gold_matrix[s2].index('R'))
                stack_tmp.append(gold_matrix[s2].index('L'))
                while stack_tmp:
                    s_tmp = stack_tmp.pop()
                    # print("计算树2节点" + str(s_tmp))
                    dis += self.node_dis(gold_tree[s_tmp], None)
                    if ('L' in gold_matrix[s_tmp] and 'R' in gold_matrix[s_tmp]):
                        stack_tmp.append(gold_matrix[s_tmp].index('R'))
                        stack_tmp.append(gold_matrix[s_tmp].index('L'))
            elif ('L' in predict_matrix[s1] and 'R' in predict_matrix[s1]) and (
                    'L' not in gold_matrix[s2] or 'R' not in gold_matrix[s2]):
                # print("计算树1节点" + str(s1)+"和它的子节点" + "计算树2节点" + str(s2))
                dis += self.node_dis(predict_tree[s1], gold_tree[s2])
                stack_tmp = []
                stack_tmp.append(predict_matrix[s1].index('R'))
                stack_tmp.append(predict_matrix[s1].index('L'))
                while stack_tmp:
                    s_tmp = stack_tmp.pop()
                    # print("计算树1节点" + str(s_tmp))
                    dis += self.node_dis(predict_tree[s_tmp], None)
                    if ('L' in predict_matrix[s_tmp] and 'R' in predict_matrix[s_tmp]):
                        stack_tmp.append(predict_matrix[s_tmp].index('R'))
                        stack_tmp.append(predict_matrix[s_tmp].index('L'))
            elif ('L' not in predict_matrix[s1] and 'R' not in predict_matrix[s1]) and (
                    'L' not in gold_matrix[s2] and 'R' not in gold_matrix[s2]):
                # print("计算树1节点" + str(s1) + "计算树2节点" + str(s2))
                dis += self.node_dis(predict_tree[s1], gold_tree[s2])
            else:
                stack1.append(predict_matrix[s1].index('R'))
                stack1.append(predict_matrix[s1].index('L'))
                stack2.append(gold_matrix[s2].index('R'))
                stack2.append(gold_matrix[s2].index('L'))
                # print("计算树1节点" + str(s1) + "计算树2节点" + str(s2))
                dis += self.node_dis(predict_tree[s1], gold_tree[s2])
        # print("final_dis",dis)
        return dis

    # 计算决策路径抽取的TP,TP+FP,TP+FN
    def decision_path(self, predict_tree, gold_tree, predict_matrix, gold_matrix):
        leaf1, leaf2, paths1, paths2 = [], [], [], []
        for i in range(len(predict_matrix)):
            if ('L' not in predict_matrix[i] and 'R' not in predict_matrix[i]):
                leaf1.append(i)
        for node in leaf1:
            path = [predict_tree[node]]
            while node != 0:
                # print(predict_matrix)
                # print(node)
                # print(predict_matrix[node])
                path.append(predict_matrix[predict_matrix[node].index('F')][node])
                path.append(predict_tree[predict_matrix[node].index('F')])
                node = predict_matrix[node].index('F')
            paths1.append(path)
        for i in range(len(gold_matrix)):
            if ('L' not in gold_matrix[i] and 'R' not in gold_matrix[i]):
                leaf2.append(i)
        for node in leaf2:
            path = [gold_tree[node]]
            while node != 0:
                path.append(gold_matrix[gold_matrix[node].index('F')][node])
                path.append(gold_tree[gold_matrix[node].index('F')])
                node = gold_matrix[node].index('F')
            paths2.append(path)
        res = 0
        for path1 in paths1:
            for path2 in paths2:
                if self.is_path_equal(path1, path2):
                    res += 1
                    break

        return res, len(paths1), len(paths2)

    # 计算三元组抽取的TP,TP+FP,TP+FN
    def triplet_extraction(self, predict_tree, gold_tree):
        predict_triplet, gold_triplet = [], []
        for i in range(len(predict_tree)):
            for triplet in predict_tree[i]["triples"]:
                predict_triplet.append(tuple(triplet))
        for i in range(len(gold_tree)):
            for triplet in gold_tree[i]["triples"]:
                gold_triplet.append(tuple(triplet))
        predict_triplet_num = len(list(set(predict_triplet)))
        gold_triplet_num = len(list(set(gold_triplet)))
        correct_triplet_num = len(list(set(gold_triplet) & set(predict_triplet)))
        return [correct_triplet_num, predict_triplet_num, gold_triplet_num]

    def node_extraction(self, predict_tree, gold_tree):
        predict_node, gold_node = [], []
        for i in range(len(predict_tree)):
            if len(predict_tree[i]['triples']) > 0:
                predict_node.append(predict_tree[i])
        for i in range(len(gold_tree)):
            if len(gold_tree[i]['triples']) > 0:
                gold_node.append(gold_tree[i])

        predict_triplet_num = len(predict_node)
        gold_triplet_num = len(gold_node)
        correct_triplet_num = 0
        for node1 in predict_node:
            for node2 in gold_node:
                if len(node1['triples']) > 0 and node1['role'] == node2['role'] and node1['logical_rel'] == node2[
                    'logical_rel'] and set(node1['triples']) == set(node2['triples']):
                    correct_triplet_num += 1
        return [correct_triplet_num, predict_triplet_num, gold_triplet_num]

    # 评测函数，共计算四个指标: 决策树的Acc；三元组抽取的F1；决策路径的F1; 树的编辑距离
    def eval(self, predict_tree, gold_tree):
        for node in predict_tree:
            for i in range(len(node['triples'])):
                node['triples'][i] = tuple(node['triples'][i])
        for node in gold_tree:
            for i in range(len(node['triples'])):
                node['triples'][i] = tuple(node['triples'][i])

        # 将符合诊疗决策树的节点前序序列转化为代表诊疗决策树结构的节点矩阵，matrix[i][j]='F'/'L'/'R'表示第j个节点是第i个节点的父/左子/右子节点
        predict_matrix = self.nodematrix(predict_tree)
        gold_matrix = self.nodematrix(gold_tree)

        # 用于计算生成树的Acc
        tree_num = (0 if predict_tree == [] else 1)
        correct_tree_num = is_tree_equal(predict_tree,gold_tree)

        # 用于计算triplet抽取的F1
        correct_triplet_num, predict_triplet_num, gold_triplet_num = self.triplet_extraction(predict_tree, gold_tree)

        # 用于计算决策路径的F1
        correct_path_num, predict_path_num, gold_path_num = self.decision_path(predict_tree, gold_tree, predict_matrix,
                                                                          gold_matrix)

        # 用于计算树的编辑距离
        edit_dis = self.edit_distance(predict_tree, gold_tree, predict_matrix, gold_matrix)

        correct_node_num, predict_node_num, gold_node_num = self.node_extraction(predict_tree, gold_tree)

        return tree_num, correct_tree_num, correct_triplet_num, predict_triplet_num, gold_triplet_num, correct_path_num, predict_path_num, gold_path_num, edit_dis, correct_node_num, predict_node_num, gold_node_num

    def tree_structure_eval(self):
        gold_tree_num, correct_tree_num = 0.000001, 0.000001
        gold_triplet_num, predict_triplet_num, correct_triplet_num = 0.000001, 0.000001, 0.000001
        gold_path_num, predict_path_num, correct_path_num = 0.000001, 0.000001, 0.000001
        gold_node_num, predict_node_num, correct_node_num = 0.000001, 0.000001, 0.000001
        edit_dis = 0
        for i in range(len(self.gold_data)):
            for node in self.pred_data[i]['tree']:
                node['logical_rel'] = 'X'
            for node in self.gold_data[i]['tree']:
                node['logical_rel'] = 'X'
            tmp = self.eval(self.pred_data[i]['tree'], self.gold_data[i]['tree'])
            gold_tree_num += tmp[0]
            correct_tree_num += tmp[1]
            correct_triplet_num += tmp[2]
            predict_triplet_num += tmp[3]
            gold_triplet_num += tmp[4]
            correct_path_num += tmp[5]
            predict_path_num += tmp[6]
            gold_path_num += tmp[7]
            edit_dis += tmp[8]
            correct_node_num += tmp[9]
            predict_node_num += tmp[10]
            gold_node_num += tmp[11]

        tree_acc = correct_tree_num / gold_tree_num
        triplet_f1 = 2 * (correct_triplet_num / predict_triplet_num) * (correct_triplet_num / gold_triplet_num) / (
                    correct_triplet_num / predict_triplet_num + correct_triplet_num / gold_triplet_num)
        path_f1 = 2 * (correct_path_num / predict_path_num) * (correct_path_num / gold_path_num) / (
                    correct_path_num / predict_path_num + correct_path_num / gold_path_num)
        tree_edit_distance = edit_dis / gold_tree_num
        node_f1 = 2 * (correct_node_num / predict_node_num) * (correct_node_num / gold_node_num) / (
                    correct_node_num / predict_node_num + correct_node_num / gold_node_num)

        return tree_acc, triplet_f1, path_f1, tree_edit_distance, node_f1
