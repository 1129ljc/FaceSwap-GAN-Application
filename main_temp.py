# @Time : 2022/1/25 17:38
# @Author : 1129ljc
# @File : main_temp.py
# @Software: PyCharm

import os
import sys
import json

from conversion import converter

model_save = {
    'obama2biden': {
        'model_dir': './model/model_3_4',
        'merge_type': 'BtoA'
    },
    'biden2obama': {
        'model_dir': './model/model_3_4',
        'merge_type': 'AtoB'
    },
    'johnson2trump': {
        'model_dir': './model/model_1_2',
        'merge_type': 'AtoB'
    },
    'trump2johnson': {
        'model_dir': './model/model_1_2',
        'merge_type': 'BtoA'
    }
}


def test(input_a, input_b, input_c):
    input_arg_task_id = input_a
    input_arg_file_path = input_b
    input_arg_ext = input_c

    input_arg_ext_json = json.loads(input_arg_ext)
    print(input_arg_ext_json)
    input_arg_ext_out_json_path = input_arg_ext_json['JSON_FILE_PATH']

    input_arg_ext_gpu_id = input_arg_ext_json['GPU_ID']

    input_arg_ext_tmp_dir = input_arg_ext_json['TMP_DIR']
    input_arg_ext_tmp_dir = os.path.join(input_arg_ext_tmp_dir, 'ljc_docs')
    input_arg_ext_out_tmp_path = os.path.join(input_arg_ext_tmp_dir, 'faceswap_gan')
    input_arg_ext_out_tmp_path = os.path.join(input_arg_ext_out_tmp_path, str(input_arg_task_id))

    input_arg_ext_base_dir = input_arg_ext_json['BASE_DIR']
    # input_arg_ext_base_dir = os.path.join(input_arg_ext_base_dir, 'ljc_docs')
    # input_arg_ext_out_base_path = os.path.join(input_arg_ext_base_dir, 'faceswap_gan')
    # input_arg_ext_out_base_path = os.path.join(input_arg_ext_out_base_path, str(input_arg_task_id))

    # if not os.path.exists(input_arg_ext_out_base_path):
    #     os.makedirs(input_arg_ext_out_base_path)
    if not os.path.exists(input_arg_ext_out_tmp_path):
        os.makedirs(input_arg_ext_out_tmp_path)

    source_file = input_arg_ext_base_dir + str(input_arg_ext_json['Source'])
    print(source_file)
    print('source_file', source_file)
    model_select = input_arg_ext_json['Model']
    if type(model_select) is list:
        model_select = model_select[0]
    assert model_select in ['obama2biden', 'biden2obama', 'johnson2trump', 'trump2johnson'], print('model select error')

    args = {
        'input': source_file,
        'output': os.path.join(input_arg_ext_out_tmp_path, 'result.mp4'),
        'face_size': 64,
        'gpu_id': input_arg_ext_gpu_id,
        'model_dir': model_save[model_select]['model_dir'],
        'merge_type': model_save[model_select]['merge_type'],
        'output_type': 1,
        'mode': 'video'
    }
    print(args)
    converter(args)

    input_file_name = args['input'].split('/')[-1]
    output_file_name = args['output'].split('/')[-1]

    result_json_content = {}
    result_json_content[input_file_name] = []
    result_json_content[input_file_name].append(
        {
            'name': '结果说明',
            'type': 'STRING',
            'value': '该视频为特定人物' + str(model_select).split('2')[0] + '的人脸视频'
        }
    )
    result_json_content[input_file_name].append(
        {
            'name': '原始视频',
            'type': str(input_file_name.split('.')[-1]),
            'value': args['input'].replace(input_arg_ext_base_dir, '')
        }
    )
    result_json_content[output_file_name] = []
    result_json_content[output_file_name].append(
        {
            'name': '结果说明',
            'type': 'STRING',
            'value': '该视频为特定人物' + str(model_select).split('2')[1] + '的伪造人脸视频，是特定人物交换模型' + str(model_select) + '的交换结果'
        }
    )
    result_json_content[output_file_name].append(
        {
            'name': '伪造视频',
            'type': str(output_file_name.split('.')[-1]),
            'value': args['output'].replace(input_arg_ext_base_dir, '')
        }
    )

    json_path = input_arg_ext_out_json_path
    with open(json_path, 'w') as f:
        json.dump(result_json_content, f)
    f.close()


if __name__ == '__main__':
    input_1 = sys.argv[1]
    input_2 = sys.argv[2]
    input_3 = sys.argv[3]
    test(input_a=input_1, input_b=input_2, input_c=input_3)
