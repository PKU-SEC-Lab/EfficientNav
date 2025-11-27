import json
import torch
import numpy as np
from typing import List, Tuple, Optional
import time
import math
import gc
import copy
from transformers.cache_utils import Cache, DynamicCache

width_weight = 0.001
gpu_node_num = 20
use_pruning = True
layer_threshold = 5

class TreeNode:
    def __init__(self, key: str, position: List[float], direction: float, waypoint = None, distance_to_parent: float = 0.0, parent: Optional['TreeNode'] = None, picture = None, describe = None):
        self.key = key
        self.position = position
        self.direction = direction
        self.waypoint = waypoint
        self.distance_to_parent = distance_to_parent
        self.picture = picture
        self.parent = parent
        self.children = []
        self.describe = describe
        self.describe_kv = []
        self.similarity = []
        self.current_inference = 0
        self.state = 'explorable'
        self.group = None

    def add_child(self, child_node: 'TreeNode'):
        child_node.parent = self
        self.children.append(child_node)

    def to_dict(self):
        return {
            'key': self.key,
            'position': self.position,
            'direction': self.direction,
            'waypoint': self.waypoint,
            'distance_to_parent': self.distance_to_parent,
            'children': [child.to_dict() for child in self.children]
        }

    @staticmethod
    def from_dict(data: dict, parent: Optional['TreeNode'] = None) -> 'TreeNode':
        node = TreeNode(
            data['key'],
            tuple(data['position']),
            data['direction'],
            data['waypoint'],
            data['distance_to_parent'],
            parent
        )
        node.children = [TreeNode.from_dict(child, node) for child in data['children']]
        return node
    
    def find_connections(self):
        connection_map = ' '
        if not self.children:
            return connection_map
        else:
            for i in range(len(self.children)):
                connection_map += f'{self.key} is connected to {self.children[i].key}. '
            for i in range(len(self.children)):
                connection_map += self.children[i].find_connections()
            return connection_map

class Navigation_map:
    def __init__(self, root: Optional[TreeNode] = None):
        self.planner_model = None
        self.semantic_model = None
        self.processor = None
        self.root = root
        self.now = root
        self.current_inference = 0
        self.num_node = 0
        self.similarity_threshould = None
        self.similarity_times = None
        self.store_in_cpu = []
        self.store_in_gpu = []
        self.store_in_gpu_score = []
        self.used_id = []
        self.device_map = None
        self.used_groups = []
        self.place_clip_id = []

    def add_node(self, parent_key: str, key: str, position: List[float], direction: float, waypoint, distance_to_parent: float,picture,describe):
        if not self.root:
            self.root = TreeNode(key, position, direction, waypoint, distance_to_parent,None,picture,describe)
            self.now = self.root
            self.now.group = 0
        else:
            parent_node = self.now
            # parent_node = self.find_node(self.root, parent_key)
            if parent_node:
                child = TreeNode(key, position, direction, waypoint, distance_to_parent, parent_node,picture,describe)
                parent_node.add_child(child)
            else:
                raise ValueError("Parent key not found in the tree")
            self.now = child
            self.get_node_group(self.planner_model,self.now)
        self.compute_kv(self.now,[])    

    def get_group_kv(self,group,node,current_node):
        group_kv = None
        cpu_flag = 0
        if node.group == group and node.key != current_node.key:
            if node.key not in self.store_in_gpu:
                self.load_kv_to_gpu(node)
                cpu_flag = 1
            group_kv = copy.deepcopy(node.describe_kv)
            if cpu_flag == 1:
                node.describe_kv = tuple((tensor[0].to('cpu'),tensor[1].to('cpu')) for i,tensor in enumerate(node.describe_kv))
            
        for child in node.children:
            group_kv_child = self.get_group_kv(group,child,current_node)
            if group_kv == None:
                group_kv = group_kv_child
            else:
                if group_kv_child is not None:
                    group_kv = tuple(tuple((torch.cat((t1[0], t2[0]), dim=2),torch.cat((t1[1], t2[1]), dim=2)))for t1, t2 in zip(group_kv,group_kv_child))
        return group_kv

    def compute_kv(self,node,num_node):
        describe = ' '
        group_kv = None
        if self.num_node > 1:
            group_kv = self.get_group_kv(node.group,self.root,node)
        for i in range(len(node.describe)):
            if i not in num_node:
                describe += node.describe[i]
        conversation_kv = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{describe}"},
                    ],
                },
            ]
        prompt_kv = self.processor.apply_chat_template(conversation_kv, add_generation_prompt=True)
        inputs_kv = self.processor(prompt_kv, padding=True, return_tensors="pt").to("cuda:0")
        # inputs_kv = processor(topomap.now.describe[i], padding=True, return_tensors="pt").to("cuda:0")
        # if group_kv is not None:
        #     import pdb;pdb.set_trace()
        with torch.no_grad():
            output_kv = self.planner_model(input_ids=inputs_kv['input_ids'],use_cache =True, past_key_values=group_kv)
        node.describe_kv = tuple(tuple((tensor[0][:,:,-1*inputs_kv['input_ids'].shape[1]:],tensor[0][:,:,-1*inputs_kv['input_ids'].shape[1]:]))for tensor in output_kv.past_key_values)
        if self.device_map == None:
            self.device_map = [[tensor[0].device,tensor[1].device] for tensor in node.describe_kv]
        if node.key not in self.store_in_gpu:
            node.describe_kv = tuple((tensor[0].to('cpu'),tensor[1].to('cpu')) for tensor in node.describe_kv)
        # import pdb;pdb.set_trace()
    
    def load_kv_to_gpu(self,node):
        node.describe_kv = tuple((tensor[0].to(self.device_map[i][0]),tensor[1].to(self.device_map[i][1])) for i,tensor in enumerate(node.describe_kv))

    def find_node(self, node: TreeNode, key: str) -> Optional[TreeNode]:
        if node.key == key:
            return node
        for child in node.children:
            found_node = self.find_node(child, key)
            if found_node:
                return found_node
        return None
    
    def find_nearest_node(self, node: TreeNode, position) -> Optional[TreeNode]:
        nearest_length = 1000
        nearest_position = None
        nearest_node = None
        for child in node.children:
            length,nearest_position_child,child_node = self.find_nearest_node(child, position)
            if length < nearest_length:
                nearest_length = length
                nearest_position = nearest_position_child
                nearest_node = child_node
        length = math.sqrt((position[0]-node.position[0])**2+(position[1]-node.position[1])**2+(position[2]-node.position[2])**2)
        if length < nearest_length:
            nearest_length = length
            nearest_position = node.position
            nearest_node = node
        return nearest_length,nearest_position,nearest_node

    def get_path(self, start_key: str, end_key: str) -> List[Tuple[str, Tuple[float, float]]]:
        start_node = self.find_node(self.root, start_key)
        end_node = self.find_node(self.root, end_key)
        if not start_node or not end_node:
            raise ValueError("One or both keys not found in the tree")
        
        # Get path from start_node to root
        path_to_root_from_start = []
        node = start_node
        while node:
            path_to_root_from_start.append(node)
            node = node.parent
        path_to_root_from_start.reverse()
        
        # Get path from end_node to root
        path_to_root_from_end = []
        node = end_node
        while node:
            path_to_root_from_end.append(node)
            node = node.parent

        # Find common ancestor
        i = 0
        while (i < len(path_to_root_from_start) and i < len(path_to_root_from_end) and
               path_to_root_from_start[i] == path_to_root_from_end[i]):
            i += 1

        common_ancestor_index = i - 1

        # Path from start to common ancestor
        path = path_to_root_from_start[:common_ancestor_index + 1]

        # Path from common ancestor to end (in reverse)
        path.extend(reversed(path_to_root_from_end[common_ancestor_index + 1:]))

        return [(node.key, node.position) for node in path]

    def save_tree(self, filename: str):
        if not self.root:
            raise ValueError("Tree is empty")
        with open(filename, 'w') as file:
            json.dump(self.root.to_dict(), file)

    def load_tree(self, filename: str):
        with open(filename, 'r') as file:
            data = json.load(file)
            self.root = TreeNode.from_dict(data)

    def print_tree(self, node: Optional[TreeNode] = None, level: int = 0):
        if node is None:
            node = self.root
        print('  ' * level + f"({node.key}) Position: {node.position}, Direction: {node.direction}, Distance to parent: {node.distance_to_parent}")
        for child in node.children:
            self.print_tree(child, level + 1)


    def create_describe(self,node,last_key,last_index,target_index):
        if use_pruning:
            selected = any(np.array([row[target_index]-width_weight*math.sqrt((node.position[0]-self.now.position[0])**2+(node.position[2]-self.now.position[2])**2) for row in node.similarity]) > self.similarity_threshould[target_index])  or any([any(np.array([row[target_index]-width_weight*math.sqrt((children.position[0]-self.now.position[0])**2+(children.position[2]-self.now.position[2])**2) for row in children.similarity]) > self.similarity_threshould[target_index] )for children in node.children])
            if node.parent != None:
                selected = selected or any(np.array([row[target_index]-width_weight*math.sqrt((node.parent.position[0]-self.now.position[0])**2+(node.parent.position[2]-self.now.position[2])**2) for row in node.parent.similarity]) > self.similarity_threshould[target_index])
        else:
            selected = True
        describe = ' '
        if selected:
            describe = f'{node.key} : '
            num_node = []
            if node.key in last_key :
                for j in range(0,len(last_key)):
                    # num_node = last_key.index(node.key)
                    if node.key == last_key[j]:
                        num_node.append(last_index[j])
            if len(num_node) < len(node.describe):
                for i in range(len(node.describe)):
                # if i!=last_index[num_node]:
                    if i not in num_node:
                        describe += node.describe[i]
            else:
                describe = ' '
        for child in node.children:
            describe += self.create_describe(child,last_key,last_index,target_index)
            
        return describe
    
    def get_similarity_threshould(self,node,last_key,last_index,target_index):
        num_node = []
        similarity_child=[]
        similarities = [0.0]
        if node.key in last_key :
            for j in range(0,len(last_key)):
                if node.key == last_key[j]:
                    num_node.append(last_index[j])
        if len(num_node) < len(node.describe):
            for i in range(len(node.describe)):
                if i not in num_node:
                    similarities.append(node.similarity[i][target_index]-width_weight*math.sqrt((node.position[0]-self.now.position[0])**2+(node.position[2]-self.now.position[2])**2))
        
        for child in node.children:
            similarity_child = self.get_similarity_threshould(child,last_key,last_index,target_index)
        for similar in similarity_child:
            similarities.append(similar)
        return similarities


    def create_describe_and_cache(self,model,node,last_key,last_index,target_index):
        num_node = []
        describe = ' '
        describe_kv = None
        if use_pruning:
            ## if node has promising objects
            selected = any(np.array([row[target_index]-width_weight*math.sqrt((node.position[0]-self.now.position[0])**2+(node.position[2]-self.now.position[2])**2) for row in node.similarity]) > self.similarity_threshould[target_index]) or node.group in self.used_groups
            ## if child node has promising objects
            selected = selected or any([any(np.array([row[target_index]-width_weight*math.sqrt((children.position[0]-self.now.position[0])**2+(children.position[2]-self.now.position[2])**2) for row in children.similarity]) > self.similarity_threshould[target_index] )for children in node.children])
            if node.parent != None:
                selected = selected or any(np.array([row[target_index]-width_weight*math.sqrt((node.parent.position[0]-self.now.position[0])**2+(node.parent.position[2]-self.now.position[2])**2) for row in node.parent.similarity]) > self.similarity_threshould[target_index])
        else:
            selected = True
        if selected:
            self.used_groups.append(node.group)
            if node.state == 'recompute':
                if node.key in last_key :
                    for j in range(0,len(last_key)):
                        if node.key == last_key[j]:
                            num_node.append(last_index[j])
                if len(num_node) >= len(node.describe):
                    node.state = 'explored'
                    if node.key in self.store_in_gpu:
                        i = self.store_in_gpu.index(node.key)
                        del self.store_in_gpu[i]
                        del self.store_in_gpu_score[i]
                else:
                    node.state = 'explorable'
                    if node.key not in self.store_in_gpu:
                        self.store_in_gpu.append(node.key)
                        self.store_in_gpu_score.append([max([node_similarities[target_index] for node_similarities in node.similarity]),node.position])
                    self.compute_kv(node,num_node)
            if node.state == 'explorable':
                for i in range(len(node.describe)):
                    if i not in num_node:
                        describe += node.describe[i]
                if node.key not in self.store_in_gpu:
                    self.store_in_gpu.append(node.key)
                    self.store_in_gpu_score.append([max([node_similarities[target_index] for node_similarities in node.similarity]),node.position])
                    self.load_kv_to_gpu(node)
                describe_kv = copy.deepcopy(node.describe_kv)
                if len(self.store_in_gpu) > gpu_node_num:
                    node_to_delete = sorted(range(len(self.store_in_gpu_score)), key=lambda x: self.store_in_gpu_score[x][0]- width_weight * math.sqrt((self.store_in_gpu_score[x][1][0]-self.now.position[0])**2+(self.store_in_gpu_score[x][1][2]-self.now.position[2])**2))[:(len(self.store_in_gpu) - gpu_node_num)]
                    node_to_delete.sort(reverse=True)
                    for i in node_to_delete:
                        del self.store_in_gpu[i]
                        del self.store_in_gpu_score[i]
            else:
                describe = ' '

        for child in node.children:
            describe_child, describe_kv_child= self.create_describe_and_cache(model,child,last_key,last_index,target_index)

            describe += describe_child
            if describe_kv == None:
                describe_kv = describe_kv_child
            else:
                if describe_kv_child is not None:
                    describe_kv = tuple(tuple((torch.cat((t1[0], t2[0]), dim=2),torch.cat((t1[1], t2[1]), dim=2)))for t1, t2 in zip(describe_kv,describe_kv_child))
       
        return describe,describe_kv


    def find_token_length(self,node,tokenizer):
        length = []
        prompt = ' '
        for i in range(len(node.describe)):
            prompt += node.describe[i]
        prompt_token = tokenizer(prompt)
        length.append(len(prompt_token['input_ids']))
        for i in range(len(node.children)):
            length_child = self.find_token_length(node.children[i],tokenizer)
            for i in range(len(length_child)):
                length.append(length_child[0])
        return length
    

    def get_node_group(self,model,node):
        threshold = 0.99
        describe,describe_kv,cache_length,cache_group = self.create_describe_and_cache_all(model,self.root)
        for i in range(len(cache_length)-1):
            cache_length[i+1] += cache_length[i]
        describe_current = ' '
        for i in range(len(node.describe)):
            describe_current += node.describe[i]
        conversation_kv = [
            {
            "role": "user",
            "content": [
                {"type": "text", "text": f"{describe_current}"},
                    ],
                },
            ]
        prompt_kv = self.processor.apply_chat_template(conversation_kv, add_generation_prompt=True)

        inputs_kv = self.processor(prompt_kv, padding=True, return_tensors="pt").to("cuda:0")
        input_ids = inputs_kv['input_ids']
        with torch.no_grad():
            if not isinstance(describe_kv, Cache):  # kept for BC (non `Cache` `past_key_values` inputs)
               describe_kv = DynamicCache.from_legacy_cache(describe_kv)
            
            hidden_states = model.language_model.model.embed_tokens(input_ids)  # 嵌入层
            past_seen_tokens = describe_kv[0][0][0][0].shape[0] if describe_kv is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + hidden_states.shape[1], device=hidden_states.device
            )
            position_ids = cache_position.unsqueeze(0)
            position_embeddings = model.language_model.model.rotary_emb(hidden_states, position_ids)
            for i,layer in enumerate(model.language_model.model.layers):
                outputs = layer(hidden_states, use_cache=True, past_key_value=describe_kv, output_attentions=True,cache_position=cache_position,position_ids=position_ids,position_embeddings=position_embeddings)
                hidden_states = outputs[0]  
                attention = outputs[1]  
                attention_sum = np.zeros(len(set(cache_group)))
                for j in range(len(cache_length)):
                    if j==0:
                        last_length = 0
                    else:
                        last_length = cache_length[j-1]
                    attention_sum[cache_group[j]] += attention[0][:,-1,last_length:cache_length[j]].cpu().numpy().sum()/(1.0*(attention[0].shape[0])) ##?
                index = np.argmax(attention_sum)
                if attention_sum[index]>threshold:
                    node.group = index
                    break
                if i > layer_threshold:
                    node.group = len(set(cache_group))
                    break

    
    def create_describe_and_cache_all(self,model,node):
        describe = ' '
        describe_kv = None
        cache_length = []
        cache_group = []
        cpu_flag = 0
        if node.group != None:
            for i in range(len(node.describe)):
                describe += node.describe[i]
            cache_length.append(node.describe_kv[0][0].shape[2])
            cache_group.append(node.group)
            if node.key not in self.store_in_gpu:
                self.load_kv_to_gpu(node)
                cpu_flag = 1
            describe_kv = node.describe_kv[:layer_threshold]
            if cpu_flag == 1:
                node.describe_kv = tuple((tensor[0].to('cpu'),tensor[1].to('cpu')) for i,tensor in enumerate(node.describe_kv))

        for child in node.children:
            describe_child, describe_kv_child,cache_length_child,cache_group_child= self.create_describe_and_cache_all(model,child)
            describe += describe_child
            cache_length.extend(cache_length_child)
            cache_group.extend(cache_group_child)
            if describe_kv == None:
                describe_kv = describe_kv_child
            else:
                if describe_kv_child is not None:
                    describe_kv = tuple(tuple((torch.cat((t1[0], t2[0]), dim=2),torch.cat((t1[1], t2[1]), dim=2)))for t1, t2 in zip(describe_kv,describe_kv_child))
        return describe,describe_kv,cache_length,cache_group