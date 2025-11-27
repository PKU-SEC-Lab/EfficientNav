
import os
import gzip
import json
import re
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration,LlamaTokenizer
import torch
import random
import habitat_sim
import habitat_sim.utils.common as utils
import numpy as np
import imageio
import matplotlib.pyplot as plt
from math import ceil
import magnum as mn
from PIL import Image as I
import quaternion
import math
import gc
import copy
from collections import namedtuple
from transformers import CLIPTokenizer, CLIPTextModel
from scipy.spatial.distance import cosine,euclidean
import datetime
current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S") 
from navigation_map import Navigation_map
from units import load_image,load_model,get_grounding_output,plot_boxes_to_image,last_non_space_char,make_cfg



robot_flag_letter = 'assistant'
max_memory={2: "47GiB",1: "47GiB",0: "47GiB"}
llava_model_path = "PATH/TO/llava-34b"
tokenizer = LlamaTokenizer.from_pretrained(llava_model_path)
processor = LlavaNextProcessor.from_pretrained(llava_model_path)
model = LlavaNextForConditionalGeneration.from_pretrained(llava_model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True,device_map="auto", max_memory=max_memory) 


config_file = "PATH/TO/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
checkpoint_path = "PATH/TO/GroundingDINO/weights/groundingdino_swint_ogc.pth"
output_dir = "images_output"
box_threshold = 0.5
text_threshold = 0.25

token_spans = None

os.makedirs(output_dir, exist_ok=True)

model_dino = load_model(config_file, checkpoint_path, cpu_only=False)

device = "cuda" if torch.cuda.is_available() else "cpu"
device0 = "cuda:0" if torch.cuda.is_available() else "cpu"
local_model_path = "PATH/TO/clip"  
tokenizer = CLIPTokenizer.from_pretrained(local_model_path)
model_clip = CLIPTextModel.from_pretrained(local_model_path).to(device0)

group_node = True ##
delete_traj = True ##
depth_threshould = 0.25
hebing_threshould = 0.001
node_pruning_num = 4
object_describe_multi_time = False ##
through_door = True ##
use_traj = False #
pay_attention_to_door = True ##
use_real_semetic = True ##
early_stop  = True #
directly_find =True ##
use_kv_cache = True
use_pruning = True

num_episode = 20
num_environment = 15
use_door_as_trajectory = False
final_goal_list = ['toilet','tv','chair','sofa','bed','plant']


def get_observation(images,depth):
    conversation_34b = [
                {
                "role": "user",
                "content": [
                    {"type": "text", "text": '''You need to make a purposeful observation of the image from the current perspective.
                    Then describe the main larger solid objects in the image in a short statement and follow the following format:
                    { "Angle": 0, "Objects": ["Object name", "Object name"] }
                    Here are some things you should be aware of:
                    1. Entrances or doorways to other Spaces in the room count as objects, which you need to describe. But do not describe doors.
                    2. Objects that are too small need no description.
                    3. You should descibe the same object only once. You can decribe 4 objects in the image at most.
                    4. Only output description follow the format, other content is not output.
                    5. Do not describe objects in the mirror. '''},
                    {"type": "image"},
                    ],
                },
    ]

    prompt = processor.apply_chat_template(conversation_34b, add_generation_prompt=True)
    inputs = processor([prompt, prompt,prompt, prompt], images, return_tensors="pt").to("cuda:0")

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=400)

    real_outputs = [processor.decode(o, skip_special_tokens=True) for o in output]

    llava_answer = []
    robot_flag = robot_flag_letter
    for real_output in real_outputs:
        answer_start = real_output.find(robot_flag)
        if answer_start != -1:
            llava_answer.append(real_output[answer_start + len(robot_flag):].strip())
        else:
            llava_answer.append(real_output.strip())

    llava_answer1 = []
    json_data = llava_answer
    num_look = 4
    position_unlooked = []
    position_looked = []
    for i in range(0,4):
        if depth[i].mean() <= depth_threshould:
            num_look -= 1
            position_unlooked.append(i*90)
            continue
        else:
            position_looked.append(i*90)
        start = json_data[i].find('{')
        end = json_data[i].rfind('}') + 1
        json_str = json_data[i][start:end]

        try:
            data = json.loads(json_str)
        except:
            break
        data["Angle"] = i * 90
        json_string = json.dumps(data, indent=4)
        if i==0:
            llava_answer1.append(json_string)
        else:
            llava_answer1.append(json_string)
    return llava_answer1,position_looked


def get_objects_boxes(llava_answer1,fig_name):
    global text_threshold
    box_info_list_sum = []
    json_objects = copy.deepcopy(llava_answer1)
    # Parse each JSON object and store in a dictionary
    angles_objects = {}
    for json_obj in json_objects:
        obj_dict = json.loads(json_obj)
        angle = obj_dict['Angle']
        objects = obj_dict['Objects']
        angles_objects[angle] = objects
    for i, (text_prompt_list, key_angle) in enumerate(zip(angles_objects.values(),angles_objects.keys())):
        image_path = f"navigation_images/{fig_name}+surroundings_angle_{key_angle}.png"
        text_prompt_list = list(set(text_prompt_list))
        text_prompt = text_prompt_list[0]
        result = ' . '.join([obj.lower() for obj in text_prompt_list]) + ' .'
        image_pil, image = load_image(image_path)
        image_pil.save(os.path.join(output_dir, f"raw_image_angle_{i}.jpg"))
        if token_spans is not None:
            text_threshold = None
            print("Using token_spans. Set the text_threshold to None.")
        boxes_filt, pred_phrases = get_grounding_output(
            model_dino, image, result, box_threshold, text_threshold, cpu_only=False, token_spans=eval(f"{token_spans}"),text_prompt=text_prompt
        )
        size = image_pil.size
        pred_dict = {
            "boxes": boxes_filt,
            "size": [size[1], size[0]],  # H,W
            "labels": pred_phrases,
        }

        image_with_box, _ , box_info_list = plot_boxes_to_image(image_pil, pred_dict)
        box_info_list_copy = copy.deepcopy(box_info_list)
        box_info_list_flag = np.zeros(len(box_info_list))
        box_info_list_real = []
        for j in range (0,len(box_info_list_copy)):
            if box_info_list_flag[j] ==1:
                continue
            if j == len(box_info_list_copy)-1:
                box_info_list_real.append(box_info_list_copy[j])
                break
            for k in range(j+1,len(box_info_list_copy)):
                if box_info_list_copy[j]['label'] == box_info_list_copy[k]['label']:
                    box_info_list_copy[j]['box'][0] = min(box_info_list_copy[j]['box'][0],box_info_list_copy[k]['box'][0])
                    box_info_list_copy[j]['box'][1] = min(box_info_list_copy[j]['box'][1],box_info_list_copy[k]['box'][1])
                    box_info_list_copy[j]['box'][2] = max(box_info_list_copy[j]['box'][2],box_info_list_copy[k]['box'][2])
                    box_info_list_copy[j]['box'][3] = max(box_info_list_copy[j]['box'][3],box_info_list_copy[k]['box'][3])   
                    box_info_list_flag[k] = 1
            box_info_list_real.append(box_info_list_copy[j])

        for j in range(0,len(text_prompt_list)):
            box_exist_flag = 0
            for k in range(0,len(box_info_list_real)):
                if box_info_list_real[k]['label'].lower() == text_prompt_list[j].lower() :
                    box_exist_flag =1
                    break
            if box_exist_flag == 0:
                box_info_list_real.append({'label': text_prompt_list[j], 'box': [0,0,1023,1023]})

        del_tmp = []
        for j in range(0,len(box_info_list_real)):
            text_exist_flag = 0
            for k in range(0,len(text_prompt_list)):
                if box_info_list_real[j]['label'].lower() == text_prompt_list[k].lower() :
                    text_exist_flag =1
                    break
            if text_exist_flag == 0:
                del_tmp.append(j)
        del_tmp.sort(reverse=True)
        for j in range(len(del_tmp)):
            del box_info_list_real[del_tmp[j]]
        box_info_list_sum.append(box_info_list_real)
        image_with_box.save(os.path.join(output_dir, f"pred_angle_{i}.jpg"))
    return box_info_list_sum


def get_objects(topomap,scene,position_looked,box_info_list_sum,semantic_observations,obj_dict):
    def get_text_embedding(text):
        inputs = tokenizer(text, return_tensors='pt').to(device0)
        with torch.no_grad():
            text_embedding = model_clip(**inputs).last_hidden_state
            text_embedding = text_embedding.mean(dim=1) 
            text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
        return text_embedding.cpu().numpy()

    def get_similarity(text1, text2):
        vec1 = get_text_embedding(text1)
        vec2 = get_text_embedding(text2)
        similarity = 1 - cosine(vec1[0], vec2[0])
        return similarity
    
    # define object
    ObjectInfo = namedtuple("ObjectInfo", ["label","angle", "obj_id", "category", "center", "sizes"])

    objects_info_filtered = []

    max_similar_objs_list = []

    agent_state = habitat_sim.AgentState()
    current_position = np.array(agent_state.position)
    empty_position = []
    for i,(angle_picture, box_info_list) in enumerate(zip(position_looked,box_info_list_sum)):
        topomap.now.similarity.append([0.0 for _ in range(len(final_goal_list))])
        empty_flag = 0
        semantic = semantic_observations[angle_picture//90]
        for box_info in box_info_list:
            label = box_info['label'].lower()
            x1, y1, x2, y2 = box_info['box']
            semantic_box = semantic[y1:y2, x1:x2]
            unique_labels = np.unique(semantic_box)
            filtered_objects = []
            for label_id in unique_labels:
                obj = scene.objects[label_id]
                object_info_filtered = ObjectInfo(
                    label=label,
                    angle=angle_picture,
                    obj_id=label_id,
                    category=obj.category.name().lower(),
                    center=obj.obb.center,
                    sizes=obj.obb.sizes
                )
                objects_info_filtered.append(object_info_filtered)
                filtered_objects.append(object_info_filtered)

            similarities = [(label,obj.angle, obj.obj_id, get_similarity(label, obj.category), obj.category, obj.center) for obj in filtered_objects]

            if similarities:
                max_similarity = max(similarities, key=lambda x: x[3])[3]
                max_similar_objs = [(label,angle, obj_id, simi, category, center) for label,angle, obj_id, simi, category, center in similarities if simi == max_similarity]
                if len(max_similar_objs) > 1:
                    closest_obj = min(max_similar_objs, key=lambda x: euclidean(current_position, x[5]))
                    max_similar_objs = [closest_obj]

                objects = []

                if len(topomap.used_id)!=0 and any(max_similar_objs[0][2] == item[0] for item in topomap.used_id):
                    item_to_remove = max_similar_objs[0][0]
                    json_origin = topomap.now.describe[i]
                    objects_angle = json.loads(json_origin)
                    objects_origin = objects_angle['Objects']
                    objects = [obj for obj in objects_origin if obj.lower() != item_to_remove.lower()]
                    obj_dict['Angle'] = angle_picture
                    obj_dict['Objects'] = objects
                    if len(objects) == 0:
                        empty_position.append(i)
                        empty_flag = 1
                        continue
                    topomap.now.describe[i] = json.dumps(obj_dict, indent=4)
                    continue
                else :
                    if not object_describe_multi_time:
                        topomap.used_id.append([max_similar_objs[0][2],max_similar_objs[0][3]])
                    if use_real_semetic:
                        json_origin = topomap.now.describe[i]
                        objects_angle = json.loads(json_origin)
                        objects_origin = objects_angle['Objects']
                        for k,obj in enumerate(objects_origin):
                            for j,similar_obj in enumerate(max_similar_objs):
                                if similar_obj[0].lower() == obj.lower():
                                    objects_angle['Objects'][k] = similar_obj[4]
                        max_similar_objs_list.append([(max_similar_objs[0][4],max_similar_objs[0][1],int(max_similar_objs[0][2]),max_similar_objs[0][3],max_similar_objs[0][4],max_similar_objs[0][5])])
                        topomap.now.describe[i] = json.dumps(objects_angle, indent=4)
                    else:
                        max_similar_objs_list.append(max_similar_objs)
                    print(max_similar_objs[0][4])
                    if use_pruning:
                        for k in range(0,len(final_goal_list)):
                            if get_similarity(final_goal_list[k], max_similar_objs[0][4]) + 0.1* max(get_similarity(final_goal_list[k], 'door'),get_similarity(final_goal_list[k], 'door frame'))> topomap.now.similarity[i][k]:
                                topomap.now.similarity[i][k] = get_similarity(final_goal_list[k], max_similar_objs[0][4])        
            else:
                max_similar_objs = []
            if empty_flag == 1:
                break
    return max_similar_objs_list,empty_position


def planning(place_describe,place_describe_cache,final_goal,trajectory):
    input_text = 'The above is a description of different places in different angles in the environment.'
    input_text+= f'Your can get to any place described in the json data. '
    input_text+= f'Your goal is to find the {final_goal}. Based on the above json data, please choose one specific object to travel to as your target. If your goal is already in the description, please choose it as the target.'
    if use_traj:
        input_text+= f'Here is the objects that you have traveled to before: {trajectory} Do not choose the objects that you have traveled to before as the target. '
    if pay_attention_to_door:
        if use_real_semetic:
            input_text+='Note that you can travel to door or door frame to other spaces if there are no clear evidence to choose the target. '
        else:
            input_text+='Note that you can travel to entrance or door frame to other spaces if there are no clear evidence to choose the target.'
    input_text+='''Return json data by referring to the following template.
            {"Place": x, "Angle": x, "Objects": ["xxxx"] }
            If your goal is already in the description, please choose it as the target. You should not output any information other than this json data. Note that your should choose only one object in one angle of one place in the json data as the target.'''
    if not use_kv_cache:
        conversation2_new_new_new = [
            {
            "role": "user",
            "content": [
                {"type": "text", "text": f"{place_describe}"},
                {"type": "text", "text": input_text},
                ],
            },
        ]
        prompt2 = processor.apply_chat_template(conversation2_new_new_new, add_generation_prompt=True)
        inputs2 = processor(prompt2, padding=True, return_tensors="pt").to("cuda:0")
        with torch.no_grad():
            output2 = model.generate(**inputs2, max_new_tokens=200)
    else:
        conma_flag = 0
        conversation2_pruning = [
            {

            "role": "user",
            "content": [
                {"type": "text", "text": input_text},
                ],
            },
        ]
        prompt_pruning = processor.apply_chat_template(conversation2_pruning, add_generation_prompt=True)
        new_input_pruning = processor(prompt_pruning, padding=True, return_tensors="pt").to("cuda:0")
        generated_tokens = []
        while True:
            with torch.no_grad():
                outputs = model(input_ids=new_input_pruning['input_ids'],
                                # attention_mask=new_inputs['attention_mask'], 
                                past_key_values=place_describe_cache, 
                                use_cache=True)
            next_token = outputs.logits.argmax(dim=-1)[:, -1:]
            generated_tokens.append(int(next_token[0][0]))
            if int(next_token[0][0]) == 7:
                print('over')
                break
            if int(next_token[0][0]) == 97 and conma_flag ==0:
                conma_flag = 1

            place_describe_cache = outputs.past_key_values
            new_input_pruning = {'input_ids': next_token}
            if len(generated_tokens)>100:
                break

    if not use_kv_cache:
        real_output2 = processor.decode(output2[0], skip_special_tokens=True)
        del output2
    else:
        real_output2 = processor.decode(generated_tokens, skip_special_tokens=True)
        if last_non_space_char(real_output2) == ']':
            real_output2 += '}'
        del generated_tokens
    torch.cuda.empty_cache()
    gc.collect()
    print(input_text)
    print(real_output2)
    
    if not use_kv_cache:
        robot_flag = 'assistant'
        answer_start = real_output2.find(robot_flag)
        if answer_start != -1:
            llava_answer2 = real_output2[answer_start + len(robot_flag):].strip()
        else:
            llava_answer2 = real_output2.strip()
    else:
        llava_answer2 = real_output2.strip()

    return llava_answer2






def val_one_episode(topomap,sim,agent,start_point,start_rotation,final_goal_id,final_goal,distance):
    if 'tv' in final_goal:
        final_goal = 'tv'

    # ==========================================================================================================================================
    # INITIAL SIM
    # =================================================================================================================================================================

    agent_state = habitat_sim.AgentState()
    agent_state.position = start_point
    agent_state.rotation = start_rotation
    agent.set_state(agent_state)
    
    # =================================================================================================================================================================
    # FIND SHORTEST PATH
    # ==========================================================================================================================================

    scene = sim.semantic_scene

    def get_object_position(object_id):
        obj = scene.objects[object_id]
        return obj.category.name(),obj.obb.center, obj.obb.sizes
    

    _,shortest_target_position, shortest_target_dims = get_object_position(final_goal_id)

    path = habitat_sim.ShortestPath()
    path.requested_start = agent.state.position
    path.requested_end = shortest_target_position

    found_path = sim.pathfinder.find_path(path)
    path_points = path.points

    shortest_length = 0
    if found_path:
        for i, point in enumerate(path_points):
            if i==0 :
                continue
            else :
                shortest_length += math.sqrt((path_points[i][0]-path_points[i-1][0])**2+(path_points[i][2]-path_points[i-1][2])**2)
    real_distance = shortest_length
    print(f'real_distance:{real_distance}')


    # ==========================================================================================================================================
    # INITIAL PARAMETERS
    # ==========================================================================================================================================


    if not use_door_as_trajectory:
        trajectory = ' '
    else:
        trajectory = 'Door. Window.'

    sub_goal_history = []
    final_length = 0


    last_target_position = agent_state.position
    ## do not navigate to the same nodes
    last_key = []
    last_angle = []
    last_index = []
    target_tuple = None
    last_answer = ' '
    for epoch in range(0,30):
        length_this_epoch = 0.0
        sr = 0
        spl = 0.0
        target_index = final_goal_list.index(final_goal)
        
        # ==========================================================================================================================================
        # GET OBSERVATION
        # ==========================================================================================================================================
        if topomap.current_inference > 0:
            nearest_length, nearest_position,nearest_node  = topomap.find_nearest_node(topomap.root,agent_state.position)
        else:
            nearest_length = 1000
        if group_node :
            skip_node = (nearest_length < hebing_threshould) and (topomap.current_inference > 0)
        else:
            skip_node = nearest_length < hebing_threshould+1 and topomap.current_inference > 0 and epoch ==0
        fig_name = f'big+{topomap.num_node}+{epoch}'
        if skip_node:
            topomap.now = nearest_node
        else:
            surroundings = []
            depth = []
            semantic_observations = []
            images = []
            images_per_row = 2
            fig, axes = plt.subplots(ceil(360 / 90 / images_per_row), images_per_row, figsize=(15, 15))

            for idx, angle in enumerate(range(0, 360, 90)):
                agent_state.rotation = habitat_sim.utils.quat_from_angle_axis(np.deg2rad(angle), np.array([0, 1.0, 0]))
                agent.set_state(agent_state)
                sur = sim.get_sensor_observations()
                surroundings.append(sur)
                semantic_observations.append(sur["semantic_sensor"])
                color_image = sur["color_sensor"]
                depth.append(sur["depth_sensor"])
                image_path = f"navigation_images/{fig_name}+surroundings_angle_{angle}.png"
                imageio.imwrite(image_path, color_image)
                row, col = divmod(idx, images_per_row)

            image1 = I.open(f"navigation_images/{fig_name}+surroundings_angle_0.png").convert("RGB")
            image2 = I.open(f"navigation_images/{fig_name}+surroundings_angle_90.png").convert("RGB")
            image3 = I.open(f"navigation_images/{fig_name}+surroundings_angle_180.png").convert("RGB")
            image4 = I.open(f"navigation_images/{fig_name}+surroundings_angle_270.png").convert("RGB")


            image_size=672
            image1 = image1.resize((image_size,image_size), I.LANCZOS)
            image2 = image2.resize((image_size,image_size), I.LANCZOS)
            image3 = image3.resize((image_size,image_size), I.LANCZOS)
            image4 = image4.resize((image_size,image_size), I.LANCZOS)
            images = [image1,image2,image3,image4]

            ##put the images into LLava for first stage, and put the output to llava for second stage, xxx is the output
            # ==========================================================================================================================================
            # DESCRIBE IMAGE
            # ==========================================================================================================================================
            
            
            llava_answer1,position_looked = get_observation(images,depth)
            json_objects = copy.deepcopy(llava_answer1)

            llava_answer1 = []

            for json_obj in json_objects:
                obj_dict = json.loads(json_obj)
                obj_dict['Objects'] = list(set(obj_dict['Objects']))
                llava_answer1.append(json.dumps(obj_dict, indent=4))

            if use_kv_cache:
                empty_position = []
                for i in range(0, len(llava_answer1)):
                    objects_angle = json.loads(llava_answer1[i])
                    obj_dict = {'Place': topomap.num_node, **{key: value for key, value in objects_angle.items()}} 
                    if len(obj_dict['Objects'])==0:
                        empty_position.append(i)
                    llava_answer1[i] = json.dumps(obj_dict, indent=4)
                    
                empty_position.sort(reverse=True)
                for i in range(len(empty_position)):
                    del llava_answer1[empty_position[i]]
                    del llava_answer1[empty_position[i]]

            if topomap.current_inference==0:
                topomap.add_node(parent_key=None, key = 'Place 0', position = copy.deepcopy(agent_state.position), distance_to_parent = 0.0, picture = images, describe = llava_answer1,direction=None,waypoint=None)
                topomap.num_node += 1
                topomap.current_inference += 1
            else:
                # topomap.add_node(parent_key=None, key = f'Place {topomap.num_node}', position = target_position, distance_to_parent = 0.0, picture = [image1,image2,image3,image4], describe = llava_answer1,direction=None,waypoint=None)
                topomap.add_node(parent_key=None, key = f'Place {topomap.num_node}', position = copy.deepcopy(agent_state.position), distance_to_parent = 0.0, picture = images, describe = llava_answer1,direction=None,waypoint=None)
                topomap.num_node += 1
                topomap.current_inference += 1

            torch.cuda.empty_cache()  
            gc.collect()

            # ==========================================================================================================================================
            # GET_OBJECTS_BOXES
            # ==========================================================================================================================================
        
            box_info_list_sum = get_objects_boxes(llava_answer1,fig_name)

            # ==========================================================================================================================================
            # GET_OBJECTS
            # ==========================================================================================================================================
            max_similar_objs_list,empty_position = get_objects(topomap,scene,position_looked,box_info_list_sum,semantic_observations,obj_dict)
            
            topomap.place_clip_id.append(max_similar_objs_list)
            
            # ==========================================================================================================================================
            # LLM决定子目标
            # ==========================================================================================================================================
            llava_answer_concat = ' '

            for i in range(0, len(topomap.now.describe)):
                llava_answer_concat += topomap.now.describe[i]

           

        if use_pruning:
            similarity = topomap.get_similarity_threshould(topomap.root,last_key,last_index,target_index)
            similarity.sort(reverse=True)
            if len(similarity) <= node_pruning_num:
                topomap.similarity_threshould[target_index] = similarity[-1]
            else:
                topomap.similarity_threshould[target_index] = similarity[node_pruning_num]
        if use_kv_cache:
            topomap.used_groups = []
            place_describe,place_describe_cache= topomap.create_describe_and_cache(model,topomap.root,last_key,last_index,target_index)
        else:
            place_describe= topomap.create_describe(topomap.root,last_key,last_index,target_index)
        print(place_describe)


        llava_answer2 = planning(place_describe,place_describe_cache,final_goal,trajectory)

        # ===================================================================================================================
        # GET SUB-GOAL
        # ===================================================================================================================

        json_data = llava_answer2
        start = json_data.find('{')
        end = json_data.rfind('}') + 1
        json_str = json_data[start:end]
        print(json_str)
        try:
            data = json.loads(json_str)
            data_tmp = int(data["Place"])
        except:
            break

        target_place = data["Place"]
        angle_goal = data["Angle"]
        objects = data["Objects"]

        last_angle = angle_goal
        target_node = topomap.find_node(topomap.root,f'Place {target_place}')
        print(f'last_key:{last_key}')
        objects_str = ', '.join(objects)
        result = f"An area of {objects_str}."
        sub_goal_history.append(objects_str)

        if (objects[0].lower() not in trajectory) and (objects[0] not in trajectory):
            trajectory += f'{objects[0]} in Place {target_place}.'

        # ==========================================================================================================================================
        # GET SUB-GOAL INFORMATION
        # ==========================================================================================================================================
        if int(target_place)>=len(topomap.place_clip_id):
            target_place = 0
        place_id = topomap.place_clip_id[target_place]
        flag_tmp = 0

        for object_tuple in place_id:
            if len(object_tuple)==0:
                continue
            if objects[0].lower() == object_tuple[0][0].lower() and angle_goal == object_tuple[0][1]:
                target_tuple = object_tuple[0]
                flag_tmp = 1
                break
            if objects[0].lower() == object_tuple[0][0].lower():
                target_tuple = object_tuple[0]
                flag_tmp = 1


        ## if the place id is wrong, check other places
        if flag_tmp ==0:
            for i,place_id in enumerate(topomap.place_clip_id):
                for object_tuple in place_id:
                    if len(object_tuple)==0:
                        continue
                    if objects[0].lower() == object_tuple[0][0].lower():
                        target_tuple = object_tuple[0]
                        flag_tmp = 1
                        target_place = i

        if flag_tmp ==0:     
            target_tuple = topomap.place_clip_id[0][0][0]
            target_place = 0
        print(target_place)

        agent_state = agent.get_state()
        topomap.now = topomap.find_node(topomap.root, f'Place {target_place}')

        if 'Place'+f' {target_place}' != topomap.now.key:
            path = habitat_sim.ShortestPath()
            path.requested_start = agent.state.position
            path.requested_end = topomap.now.position

            found_path = sim.pathfinder.find_path(path)
            path_points = path.points

            if found_path:
                for i, point in enumerate(path_points):
                    if i==0 :
                        continue
                    else :
                        final_length += math.sqrt((path_points[i][0]-path_points[i-1][0])**2+(path_points[i][2]-path_points[i-1][2])**2)
                        length_this_epoch += math.sqrt((path_points[i][0]-path_points[i-1][0])**2+(path_points[i][2]-path_points[i-1][2])**2)

        observations = []
        semantic_observations = []
        topomap.now.position[1]=0.0
        agent_state.position = topomap.now.position
        agent_state.rotation = utils.quat_from_angle_axis(np.deg2rad(angle_goal), np.array([0, 1.0, 0]))
        agent.set_state(agent_state)
        obs = sim.get_sensor_observations()
        observations.append(obs)
        semantic_observations.append(obs["semantic_sensor"])
        color_image = obs["color_sensor"]

        scene = sim.semantic_scene

        # ==========================================================================================================================================
        # FIND PATH
        # ==========================================================================================================================================

        last_angle = target_tuple[1]
        target_node = topomap.find_node(topomap.root,f'Place {target_place}')
        if delete_traj:
            for i in range(0,len(target_node.describe)):
                last_data = json.loads(target_node.describe[i])
                if last_angle == last_data["Angle"]:
                    last_key.append(f'Place {target_place}')
                    last_index.append(i)
                    target_node.state = 'recompute'
                    break

        print(f'final_goal:{final_goal},sub_goal:{target_tuple[0]},place:{target_place},trajectory:{trajectory}',)
        sub_target_id = target_tuple[2]

        def is_door(object_id):
            obj = scene.objects[object_id]
            return obj.category.name() == "door" or obj.category.name() == "door frame"
        
        if directly_find and epoch == 29:
            for i,place_id_tmp in enumerate(topomap.place_clip_id):
                for j,object_tmp in enumerate(place_id_tmp):
                    if object_tmp[0][0].lower() in final_goal.lower() or final_goal.lower() in object_tmp[0][0].lower() or (final_goal == 'sofa' and 'couch' in object_tmp[0][0].lower()) or (final_goal == 'tv' and 'television' in object_tmp[0][0].lower()):
                            sub_target_id = object_tmp[0][2]

        def quaternion_to_direction(quat):
            forward = np.array([0, 0, -1]) 
            rotated_direction = quaternion.as_rotation_matrix(quat) @ forward
            return rotated_direction

        def detect_distance_ahead(agent_position, direction, step_size=0.25, max_distance=5.0):
            distance_traveled = 0.0
            current_position = np.array(agent_position)
            while distance_traveled < max_distance:
                next_position = current_position + direction * step_size
                if not sim.pathfinder.is_navigable(next_position):
                    break
                current_position = next_position
                distance_traveled += step_size
            return distance_traveled
        

        
        
        print(f'sub_target_id:{sub_target_id}')
        target_name,target_position, target_dims = get_object_position(sub_target_id)
        print(f'target:{target_tuple[4].lower()}')
        print(f'final_goal:{final_goal.lower()}')
        print(f'final_length:{final_length}')

        current_place = agent_state.position
        if target_position[0] == last_target_position[0] and target_position[2] == last_target_position[2]:
            if objects[0] in trajectory:
                continue
            else:
                trajectory += f'{objects[0]}. '
                continue
        last_target_position = copy.deepcopy(target_position)
        if target_position is not None:
            path = habitat_sim.ShortestPath()
            path.requested_start = agent.state.position
            path.requested_end = target_position

            current_position = copy.deepcopy(agent.state.position)
            previous_position = copy.deepcopy(current_position)
            steps = 0
            total_distance_traveled = 0.0
            step_size = 0.25
            current_index = 0

            found_path = sim.pathfinder.find_path(path)
            path_points = path.points
            if found_path:
                observations = []
                while current_index < len(path_points) - 1:
                    segment_start = current_position
                    segment_end = np.array(path_points[current_index + 1])  # 确保 segment_end 可写

                    direction = segment_end - segment_start
                    segment_distance = np.linalg.norm(direction)
                    if segment_distance <= step_size:
                        current_position = segment_end
                        current_index += 1
                    else:
                        direction /= segment_distance
                        current_position += direction * step_size

                    distance_to_target = np.linalg.norm(current_position - target_position)
                    if early_stop ==True:
                        stop_distance = 0.25
                    else:
                        stop_distance = 0.05
                    if distance_to_target <= stop_distance :
                        print("Agent is within 1m of the target. Stopping.")
                        break

                    agent_state = habitat_sim.AgentState()
                    agent_state.position = current_position

                    if current_index < len(path_points) - 1:
                        next_point = np.array(path_points[current_index + 1])  # 确保 next_point 可写
                        direction_to_next = next_point - current_position
                        direction_to_next /= np.linalg.norm(direction_to_next)

                        tangent_orientation_matrix = mn.Matrix4.look_at(
                            current_position, current_position + direction_to_next, np.array([0, 1.0, 0])
                        )
                        if np.any(np.isnan(tangent_orientation_matrix)):
                            continue
                        tangent_orientation_q = mn.Quaternion.from_matrix(
                            tangent_orientation_matrix.rotation()
                        )
                        agent_state.rotation = utils.quat_from_magnum(tangent_orientation_q)
                    
                    agent.set_state(agent_state)

                    step_distance = np.linalg.norm(current_position - previous_position)
                    total_distance_traveled += step_distance
                    previous_position = current_position.copy()

                    observations = sim.get_sensor_observations()
                    color_image = observations["color_sensor"]
                    image_path = f"tmp/navigation_images/navigation_step_{steps + 1}.png"
                    imageio.imwrite(image_path, color_image)
                    steps += 1

                if is_door(sub_target_id) and through_door:
                    print("Target object is a door. Detecting distance ahead...")
                    direction_to_target = quaternion_to_direction(agent.state.rotation)
                    max_distance_ahead = detect_distance_ahead(agent.state.position, direction_to_target)
                    print(f"Maximum distance ahead in the current direction: {max_distance_ahead:.2f} meters")
                    move_steps = int(max_distance_ahead / 0.25)
                    if move_steps > 0:
                        for _ in range(move_steps+3): 
                            agent.act("move_forward")
                            total_distance_traveled += 0.25
                final_length += total_distance_traveled
                length_this_epoch += total_distance_traveled
            else:
                print('No path found')
        if not(last_answer == llava_answer2 and use_pruning):
            last_answer = llava_answer2


        current_place = agent_state.position
        if target_name in final_goal or final_goal in target_name or (final_goal == 'sofa' and 'couch' in target_name.lower()) or (final_goal == 'tv' and 'television' in target_name.lower()) or math.sqrt((current_place[0]-shortest_target_position[0])**2+(current_place[2]-shortest_target_position[2])**2) < 1.0:
            sr = 1
            if final_length == 0:
                spl = 1
            else:
                spl = distance / final_length
            if spl > 1:
                spl = 1

        if final_goal.lower() in target_tuple[4].lower()  or (final_goal == 'sofa' and 'couch' in target_tuple[4].lower()) or (final_goal == 'tv' and 'television' in target_tuple[4].lower()):
            break

    return sr,spl,real_distance,final_length




def val_auto():
    config_file_name = 'data/VAL/content'
    file_name = 'data/VAL'

    files = os.listdir(config_file_name)

    for i, file in enumerate(files):
        SR = 0.0
        SPL = 0.0
        total_episode = 0
        total_length = 0.0
        total_length_sr = 0.0
        easy_SR = 0.0
        easy_SPL = 0.0
        easy_episode = 0
        easy_threshould = 6.0
        easy_length = 0.0
        medium_SR = 0.0
        medium_SPL = 0.0
        medium_episode = 0
        medium_length = 0.0
        hard_threshould = 9.0
        hard_SR = 0.0
        hard_SPL = 0.0
        hard_episode = 0
        hard_length = 0.0
        if i >= num_episode:
            break
        config_file_path = os.path.join(config_file_name, file)
        
        with gzip.open(config_file_path, 'rt', encoding='utf-8') as file:
            data = json.load(file)
            episodes = data.get('episodes', [])
            scene_id = episodes[0]['scene_id']
            match = re.search(r'val/([^/]+)/', scene_id)
            if match:
                extracted_part = match.group(1)
            else:
                print("can not find")
        extracted_name_part = extracted_part[6:]
        file_path = os.path.join(file_name, extracted_part)

        sim_settings = {
            "width": 1024,
            "height": 1024,
            "scene": os.path.join(file_path,extracted_name_part+'.basis.glb'),
            "scene_dataset": "data/VAL/hm3d_annotated_val_basis.scene_dataset_config.json",
            "default_agent": 0,
            "sensor_height": 1,
            "color_sensor": True,
            "depth_sensor": True,
            "semantic_sensor": True,
            "seed": 7,
            "enable_physics": False,
            "fov_horizontal": 90.0  
        }


        cfg = make_cfg(sim_settings)

        try:
            sim.close()
        except NameError:
            pass

        sim = habitat_sim.Simulator(cfg)

        random.seed(sim_settings["seed"])
        sim.seed(sim_settings["seed"])

        sim.initialize_agent(agent_id=0)
        agent = sim.agents[0]

        try: 
            topo_map
            del topo_map
            gc.collect()
            print('scene change')
        except:
            pass

        topomap = Navigation_map()
        topomap.planner_model = model
        topomap.semantic_model = model_clip
        topomap.processor = processor 
        topomap.similarity_threshould = [0.0 for _ in range(len(final_goal_list))]
        topomap.similarity_times = [0 for _ in range(len(final_goal_list))]

        SR_eposion = []
        fail_eposion = []
        subgoal_found = []

        for j,episode in enumerate(episodes):
            if j >= num_environment:
                break
            start_point = episode['start_position']
            start_rotation = episode['start_rotation']
            euclidean_distance = episode['info']['euclidean_distance']
            geodesic_distance = episode['info']['geodesic_distance']
            final_goal_id = episode['info']['closest_goal_object_id']
            final_goal = episode['object_category']

            distance = geodesic_distance
            sr, spl, real_distance,final_length= val_one_episode(topomap,sim,agent,start_point,start_rotation,final_goal_id,final_goal,distance)
            SR += sr
            SPL += spl
            total_episode +=1
            total_length += final_length
            if sr == 1:
                total_length_sr += final_length
                SR_eposion.append(j)
                subgoal_found.append(final_goal)
            else:
                fail_eposion.append(final_goal)
            if distance < easy_threshould:
                easy_SR += sr
                easy_SPL += spl
                easy_episode +=1
                if sr == 1:
                    easy_length += final_length
            elif distance > hard_threshould:
                hard_SR += sr
                hard_SPL += spl
                hard_episode +=1
                if sr == 1:
                    hard_length += final_length
            else:
                medium_SR += sr
                medium_SPL += spl
                medium_episode += 1
                if sr == 1:
                    medium_length += final_length
            os.makedirs(f'output/{current_time}', exist_ok=True)
            file_name_result = f'output/{current_time}/results{i}_test.txt'
            with open(file_name_result, 'w') as file:
                file.write(f"SR: {SR}\n")
                file.write(f"SPL: {SPL}\n")
                file.write(f"Total Episodes: {total_episode}\n")
                file.write(f"Total Length: {total_length}\n")
                file.write(f"Easy SR: {easy_SR}\n")
                file.write(f"Easy SPL: {easy_SPL}\n")
                file.write(f"Easy Episodes: {easy_episode}\n")
                file.write(f"Easy Length: {easy_length}\n")
                file.write(f"Medium SR: {medium_SR}\n")
                file.write(f"Medium SPL: {medium_SPL}\n")
                file.write(f"Medium Episodes: {medium_episode}\n")
                file.write(f"Medium Length: {medium_length}\n")
                file.write(f"Hard SR: {hard_SR}\n")
                file.write(f"Hard SPL: {hard_SPL}\n")
                file.write(f"Hard Episodes: {hard_episode}\n")
                file.write(f"Hard Length: {hard_length}\n")
                file.write(f"SR eposion: {SR_eposion}\n")
                file.write(f"SR subgoal: {subgoal_found}\n")
                file.write(f"num_node: {topomap.num_node}\n")





val_auto()