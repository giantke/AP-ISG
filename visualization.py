import json
import networkx as nx
import ipycytoscape
import ipywidgets as widgets

import pandas as pd
import numpy as np
import os, re
from numpy import genfromtxt
import json
import cv2
# import pydicom
import matplotlib
import matplotlib.pyplot as plt
print(matplotlib.__version__)

import pandas as pd
import ijson
import pdb


# TODO: set paths for images_dir and graph_dir (scene graphs) to how things are set up on your local machine

# Here we flattened the nested directories from MIMIC-CXR to just one directory for all the dicoms.
# images_dir = '/data/MIMIC/images/'
images_dir = '/mnt/wsl/PHYSICALDRIVE3p1/datasets/mimic-cxr-jpg/files/p10/p10001122/s53447138/'

# Point to path where you unzipped the chest imagenome dataset
# graph_dir = '../../../subset/scene_graph/'
graph_dir = '/mnt/wsl/PHYSICALDRIVE3p1/chest-imagenome-dataset-1.0.0/silver_dataset/scene_graph'

# specify the image you want to visualize
image_id = '07b9ddda-9a4a1e1a-4495463d-4c77d947-ed368713.dcm'



def readJSON(filepath):
    try:
        with open(filepath) as f:
            data = json.load(f)
            return data
    except Exception as e:
        print('File does not exist', filepath)
        return None


# Read relevant parts for the scene graph json
# Rearrange so can look things up by object_id or bbox_name
def readSceneGraph(image_id, basepath):
    filepath = os.path.join(basepath, str(image_id.replace('.dcm', '')) + '_SceneGraph.json')
    print(filepath)
    # pdb.set_trace()
    data = readJSON(filepath)
    reason = data['reason_for_exam']
    objects = dict()
    objs = data['objects']
    for obj in objs:
        key = obj['object_id']
        objects[key] = obj
    attributes = dict()
    attrs = data['attributes']
    for attr in attrs:
        key = attr['bbox_name']
        if attr[key]:
            attributes[key] = attr
    relations = data['relationships']
    return reason, objects, attributes, relations


# load dicom image given path to images dir and dicom_id
def load_dicom(image_id, original_folder_images='/data/MIMIC/images/'):
    ds = pydicom.dcmread(os.path.join(original_folder_images, image_id)).pixel_array.astype(np.float)
    ds -= np.min(ds)
    ds /= np.max(ds)
    ds *= 255
    return ds


# Resize and pad image
def resize_pad(image, width, return_ratio=False):
    old_size = image.shape[:2]  # old_size is in (height, width) format

    ratio = float(width) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    # new_size should be in (width, height) format

    im = cv2.resize(image, (new_size[1], new_size[0]), interpolation=cv2.INTER_NEAREST)

    delta_w = width - new_size[1]
    delta_h = width - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                value=color)

    if return_ratio:
        return new_im, top, bottom, left, right, ratio
    else:
        return new_im


def checkCoord(x, dim):
    if x < 0:
        x = 0
    if x > dim:
        x = dim
    return x


# Draw bbox on image -- has the option to save image with not margin for annotation
def drawBbox(image, x1, y1, x2, y2, plot=True, exportfilepath=None):
    image = image.copy()
    stroke = 10
    w, h = image.shape
    x1 = checkCoord(int(x1), w)
    y1 = checkCoord(int(y1), h)
    x2 = checkCoord(int(x2), w)
    y2 = checkCoord(int(y2), h)
    image[y1:y1 + stroke, x1:x2] = 3
    image[y2:y2 + stroke, x1:x2] = 3
    image[y1:y2, x1:x1 + stroke] = 3
    image[y1:y2, x2:x2 + stroke] = 3

    if plot:
        dpi = 80
        # What size does the figure need to be in inches to fit the image?
        height, width = image.shape
        figsize = width / float(dpi), height / float(dpi)
        figsize = width / 500., height / 500.
        # To make a figure without the frame :
        #         fig = plt.figure(frameon=False) # gives a bug in the plt 3.1.0 + this jupyter notebook env for some reason
        fig = plt.figure()
        fig.set_size_inches(figsize)
        # To make the content fill the whole figure
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        # Then draw your image on it :
        ax.imshow(image, cmap='gray', aspect='auto', interpolation='nearest')

        # Ensure we're displaying with square pixels and the right extent.
        # This is optional if you haven't called `plot` or anything else that might
        # change the limits/aspect.
        ax.set(xlim=[-0.5, width - 0.5], ylim=[height - 0.5, -0.5], aspect='auto')

    if exportfilepath != None:
        fig.savefig(exportfilepath, dpi=dpi, transparent=True)

    #     if plot:
    #         plt.figure(figsize=(10, 10))
    #         plt.imshow(image, cmap='gray')
    #         # plt.close() #or won't show up in notebook if run
    #     if exportfilepath!=None:
    #         fig.savefig(exportfilepath, dpi=80, transparent=True)
    return image


# visualize an object given image_id and name
def plot_object(image, objects, bbox_name, image_id, original):
    object_id = image_id.replace('.dcm', '') + '_' + bbox_name
    if object_id in objects:
        box = objects[object_id]
        if original:
            x1 = box['original_x1']
            y1 = box['original_y1']
            x2 = box['original_x2']
            y2 = box['original_y2']
        else:
            x1 = box['x1']
            y1 = box['y1']
            x2 = box['x2']
            y2 = box['y2']
        image_plotted = drawBbox(image, x1, y1, x2, y2)
        return image_plotted
    else:
        print('Object ', bbox_name, ' not extracted for this image.')
        return image





# Read the bounding box coordinates (objects) for different objects for this scene/CXR image
reason, objects, attributes, __ = readSceneGraph(image_id, graph_dir)
# print(reason)
# print()
# print(objects)
# print()
# print(attributes)
# pdb.set_trace()


class PDF(object):
    def __init__(self, pdf, size=(200,200)):
        self.pdf = pdf
        self.size = size

    def _repr_html_(self):
        return '<iframe src={0} width={1[0]} height={1[1]}></iframe>'.format(self.pdf, self.size)

    def _repr_latex_(self):
        return r'\includegraphics[width=1.0\textwidth]{{{0}}}'.format(self.pdf)

# PDF('cxr_knowledge_graph_neo4j.pdf',size=(900,600))


## Load enriched RDF graph

## Use 'scenegraph_postprocessing.py' in the utils/postprocessing/ directory to generate this simpler json from the more detailed scene graph jsons
# TODO json文件太大了，无法load，直接kill
# with open('/mnt/wsl/PHYSICALDRIVE3p1/chest-imagenome-dataset-1.0.0/silver_dataset/study_level_attribute_rdfgraphs.json','r') as f:
#     data = json.load(f)


# This is the mapping file -- note that only AP and PA view images are included in the Chest ImaGenome dataset
records = pd.read_csv('/mnt/wsl/PHYSICALDRIVE3p1/chest-imagenome-dataset-1.0.0/utils/cxr-record-list_view.csv')
records = records[records.ViewPosition.isin(['PA','AP'])].reset_index(drop=True).copy()
print(records.shape)
print(records.head())



study_id = records[records.dicom_id == image_id.replace('.dcm','')].study_id
# print("study_id", str(study_id))
print("study_id", str(study_id).split()[1])
study_id = str(study_id).split()[1]
# pdb.set_trace()


# with open('/mnt/wsl/PHYSICALDRIVE3p1/chest-imagenome-dataset-1.0.0/silver_dataset/study_level_attribute_rdfgraphs.json','r') as f:
#     parser = ijson.parse(f)
#
#     # 当前处理的键
#     current_key = None
#
#     # 标志以指示是否正在处理 "study_id" 键
#     is_study_id = False
#
#     # 用于存储匹配的 JSON 结构
#     matching_jsons = []
#
#     # 使用迭代器遍历 JSON 数据
#     for prefix, event, value in parser:
#         if event == 'map_key':
#             # 更新当前处理的键
#             current_key = value
#             # 检查是否是 "study_id" 键
#             if current_key == study_id:
#                 is_study_id = True
#                 matching_json = None
#             else:
#                 is_study_id = False
#         elif is_study_id:
#             # 如果当前正在处理 "study_id" 键，将整个 JSON 结构添加到匹配的 JSON 结构列表中
#             if current_key == study_id:
#                 matching_jsons.append(value)
#
#     # 打印匹配的 JSON 结构
#     for matching_json in matching_jsons:
#         print(matching_json)
# # 如果需要，你可以将匹配的 JSON 结构保存到文件中，例如 JSON 文件、文本文件等
# with open('output.json', 'w') as output_file:
#     json.dump(matching_json, output_file, indent=4)





## Choose one graph sample to visualize
# study_id = '58400371'
# sample = data[study_id].copy()

with open('study_id.json', 'r') as f:
    j = json.load(f)
    for key, value in j.items():
        if key == study_id:
            sample = value.copy()
            break
print(sample)
# pdb.set_trace()
# The RDF graph from the processing still misses the clinical history - reason for exam
# which can be read from the scene graph json
print(reason)

# Add the reason for exam relation to the graph
reason_rel = [[reason, 'history'],[study_id,'study_id'],'reason_for_exam']

sample.append(reason_rel)

# The enriched RDF format for the nodes and edges

# # with this pattern:
# [
#        [ [node_id_1, node_type_1], [node_id_2, node_type_2], relation_name_A ],
#        [ [node_id_1, node_type_1], [node_id_3, node_type_3], relation_name_B ],
#        ...
# ]

# The UMLS CUI for each node is '|' concatenated with the node name.
# If there are more than one suitable UMLS CUI, then they are joined by ;;

print(sample)


# TODO  Visualize the graph sample - directed

## Generating graph from sample

class CustomNode(ipycytoscape.Node):
    def __init__(self, name, node_type=''):
        super().__init__()
        self.data['id'] = name
        self.classes = node_type

G = nx.DiGraph()
for edge in sample:
    node1 = edge[0]
    node2 = edge[1]
    n1 = CustomNode(node1[0].split('|')[0], node_type=node1[1])
    n2 = CustomNode(node2[0].split('|')[0], node_type=node2[1])
    G.add_node(n1)
    G.add_node(n2)
    # negated
    if 'NOT' in edge[2]:
        value = 1
    # affirmed
    elif 'annotation' in edge[2]:
        value = 0
    # parent to child
    elif 'p2c' in edge[2]:
        value = 2
    # child to parent
    elif 'c2p' in edge[2]:
        value = 3
    # other
    else:
        value = 4
    G.add_edge(n1, n2, rel_type=edge[2], negated=value)

## Visualize the graph sample - directed
vis_G_directed = ipycytoscape.CytoscapeWidget()
vis_G_directed.graph.add_graph_from_networkx(G)
vis_G_directed.set_layout(nodeSpacing=15, edgeLengthVal=15)
vis_G_directed.set_style([
                    {'selector': 'node','style': {
                        'font-family': 'helvetica',
                        'font-size': '12px',
                        'label': 'data(id)'
                        }
                    },
                    {
                        'selector': 'node.history',
                        'css': {
                            'background-color': 'pink'
                        }
                    },
                    {
                        'selector': 'node.location',
                        'css': {
                            'background-color': 'blue'
                        }
                    },
                    {
                        'selector': 'node.study_id',
                        'css': {
                            'background-color': 'black'
                        }
                    },
                    {
                        'selector': 'node.nlp_annotation',
                        'css': {
                            'background-color': 'red'
                        }
                    },
                    {
                        'selector': 'node.anatomicalfinding_annotation',
                        'css': {
                            'background-color': 'orange'
                        }
                    },
                    {
                        'selector': 'node.technicalassessment_annotation',
                        'css': {
                            'background-color': 'purple'
                        }
                    },
                    {
                        'selector': 'node.tubesandlines_annotation',
                        'css': {
                            'background-color': 'green'
                        }
                    },
                    {
                        'selector': 'node.device_annotation',
                        'css': {
                            'background-color': 'green'
                        }
                    },
                    {
                        'selector': 'node.disease_annotation',
                        'css': {
                            'width': 2,
                            'background-color': 'yellow'
                        }
                    },
                    {
                        "selector": "edge[negated=0]",
                        "style": {
                            'width': 2,
                            "line-color": "crimson"
                        }
                    },
                    {
                        "selector": "edge[negated=1]",
                        "style": {
                            'width': 2,
                            "line-color": "lightgreen"
                        }
                    },
                    {
                        'selector': 'edge',
                        'style': {
                            'width': 2,
                            'target-arrow-shape': 'triangle',
                            'curve-style': 'bezier'
                        }
                    }
                  ])


# Edges are now directed and color coded for different relation types

# Affirmed edges are red
# Negated edges are light green
# other edges are bezier (light grey)

vis_G_directed

# from PIL import Image
# from io import BytesIO
#
#
# # 在 Jupyter Notebook 中显示可视化
# display(vis_G_directed)
#
# # 截取可视化并保存为图像
# screenshot = widgets.screenshot(widget=vis_G_directed, height=400, width=600)
# screenshot_image = Image(value=screenshot.data, format='png')
#
#
# # 保存图像到文件（替换为你的文件路径）
# with open('cytoscape_image.png', 'wb') as f:
#     f.write(screenshot_image.data)
