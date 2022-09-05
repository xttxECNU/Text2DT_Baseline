import numpy as np


class TreeSoftDecoder():
    def __init__(self, pred_result, node_separate_threshold):
        self.pred_result = pred_result
        self.node_separate_threshold = node_separate_threshold

    def get_node_pre(self,triple_divide,i,j,matrix):
        #获得节点表中第i行第j列的概率分布
        num = 0
        result = np.array([0,0,0,0])
        for index1 in triple_divide[i]:
            for index2 in triple_divide[j]:
                result = np.add(matrix[index1][index2],result)
                num = num+1
        return result/num

    def triple_to_node(self, cos_martix, num_triple, probability_martix):
        node_flag = [True]*num_triple
        triple_divide = []

        while True in node_flag:
            #将同一节点中的三元组划分到一起
            index = node_flag.index(True)
            node_flag[index] = False
            tmp = [index]
            for i in range(num_triple):
                if cos_martix[index][i]> self.node_separate_threshold and node_flag[i] == True:
                    tmp.append(i)
                    node_flag[i] = False
            triple_divide.append(tmp)

        #获得节点对表
        probability_martix = probability_martix.reshape(num_triple,num_triple,4)
        len_node = len(triple_divide)
        node_matrix = np.zeros([len_node,len_node,4])
        for i in range(len_node):
            for j in range(len_node):
                node_matrix[i][j] = self.get_node_pre(triple_divide,i,j,probability_martix)
        return(node_matrix,triple_divide)

    def find_root(self, node_matrix):
        node_len = len(node_matrix)
        scores = [0] * node_len
        for i in range(node_len):
            # 由于一个节点最多和其他三个节点产生关系，选择最后可能产生关系的三个节点（及label为N概率最低的三个节点）,根据这三个节点计算该节点为根节点的分数
            score = 0
            node_row = node_matrix[i][np.argsort(node_matrix[i][:, 3])]
            for j in range(len(node_row)):
            #一个节点是其他节点的父亲的概率越高，是其他节点的孩子的概率越低，该节点为根节点的分数越高
                score += node_row[j][2] - node_row[j][1] - node_row[j][0]
            scores[i] = score
        scores = np.array(scores)
        return scores.argmax()

    def findchildren(self, cur,node_matrix,node_flag):
        left = "NIL"
        right = "NIL"
        node_len = len(node_flag)
        node_row = node_matrix[cur, :]

        # 选择父亲节点对应行label为F概率最大的两个节点为候选子节点
        child_sort = np.argsort(node_row[:, 2])[::-1]
        child_candidate = []
        for i in range(node_len):
            if node_flag[child_sort[i]] == True: #该节点没有在树中确定位置
                child_candidate.append(child_sort[i])
                if len(child_candidate) == 2:
                    break

        if len(child_candidate) == 1:  #一个启发式规则：如果一个节点只有一个孩子，该孩子为左子节点
            label = node_matrix[child_candidate[0]][cur].argmax()
            if label == 0:
                left = child_candidate[0]
            if label == 1:
                left = child_candidate[0]

        if len(child_candidate) == 2:
            label0 = node_matrix[child_candidate[0]][cur].argmax()
            label1 = node_matrix[child_candidate[1]][cur].argmax()
            #如果两个子节点均为左孩子或右孩子，则从中选一个
            if label0 == 0:
                left = child_candidate[0]
            if label0 == 1:
                right = child_candidate[0]
            if label1 == 0:
                left = child_candidate[1]
            if label1 == 1:
                right = child_candidate[1]

        return left, right

    def node_to_tree(self, node_matrix):
        #从节点对表中解码出树
        node_len = len(node_matrix)
        #寻找根节点
        root = self.find_root(node_matrix)

        tree=[]
        stack=[root]

        #标记已经确定位置的节点，位置确定的节点将不参与后续树的生成
        node_flag = [True]*node_len
        node_flag[root] =False

        while stack:
            #深度优先生成树
            cur = stack.pop()  # pop()方法移除list中的最后一个元素，并返回元素的值
            if cur =="nonenode":
                #该节点为补充的没有实际意义的节点
                tree.append({"role": 'D', "triples": 'none', "logical_rel": "null"})
            else:
                #找到当前节点的孩子
                left, right=self.findchildren(cur,node_matrix,node_flag)
                if left =="NIL" and  right =='NIL':
                    tree.append({"role": 'D', "triples": cur, "logical_rel": "null"})
                if left != "NIL" and right != 'NIL':
                    stack.append(right)
                    stack.append(left)
                    node_flag[right] = False
                    node_flag[left] = False
                    tree.append({"role": 'C', "triples": cur, "logical_rel": "null"})
                if left != "NIL" and right == 'NIL':
                    # 右节点为空，补充无实义节点
                    stack.append("nonenode")
                    stack.append(left)
                    node_flag[left] = False
                    tree.append({"role": 'C', "triples": cur, "logical_rel": "null"})
                if left == "NIL" and right != 'NIL':
                    # 左节点为空，补充无实义节点
                    stack.append(right)
                    node_flag[right] = False
                    stack.append("nonenode")
                    tree.append({"role": 'C', "triples": cur, "logical_rel": "null"})

        return tree

    def logic_pre(self, triples):
        #为了训练tree_decoder,将所有对logical_rel设置为'null'，在tree_decoder对评估文件中也将ground_truth的logical_rel设置为'null'
        return "null"

    def cos_dis(self, a, b):
        #计算相似度
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def softdecoder(self):
        trees=[]
        for i in range(len(self.pred_result['tail_entitys_to_index'])):
            cos_martix=[]
            num_triple = len(self.pred_result['tail_entitys_to_index'][i][0])
            entity2triple = self.pred_result['tail_entitys_to_index'][i][0]

            # 如果没有三元组，返回空树
            if num_triple==0:
                trees.append({"text": '', "tree": [{"role": 'D', "triples": [], "logical_rel": "null"}]})
                continue

            probability_martix=np.array(self.pred_result['martix'][i]).reshape(num_triple,num_triple*4)

            # 计算每行之间的相似度
            for j in range(num_triple):
                for k in range(num_triple):
                    cos_martix.append(self.cos_dis(probability_martix[j],probability_martix[k]))
            cos_martix = np.array(cos_martix).reshape(num_triple,num_triple)

            # 得到节点对表
            node_matrix, triple_divide = self.triple_to_node(cos_martix, num_triple, probability_martix)

            # 解码树
            tree = self.node_to_tree(node_matrix)

            #将三元组索引转换为三元组
            for node in tree:
                tmp=[]
                map_index=node['triples']
                if map_index !='none' :
                    for j in triple_divide[map_index]:
                        tmp.append(entity2triple[j][2])
                    node['triples'] = tmp
                    #判断三元组间对逻辑关系
                    node['logical_rel'] = self.logic_pre(node['triples'])
                else:
                    node['triples']=[]
                    node['logical_rel'] = 'null'

            trees.append({"text":'',"tree":tree})
        return trees