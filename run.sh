#!/usr/bin/env bash
task_id="$1"
file_path="$2"
ext_cont="$3"
echo "$task_id"
echo "$file_path"
echo "$ext_cont"
source activate faceswap_gan
python /home/sklois/soft/ljc_methods/faceswap_gan/main.py "$task_id" "$file_path" "$ext_cont"
conda deactivate
# bash /home/sklois/soft/ljc_methods/faceswap_gan/run.sh 001 /home/sklois/soft/ljc_methods/faceswap_gan/test_video/ '{"JSON_FILE_PATH": "/home/sklois/soft/ljc_methods/faceswap_gan/temp/call_faceswap_gan.json", "TMP_DIR": "/home/sklois/soft/ljc_methods/faceswap_gan/temp/", "BASE_DIR": "/home/sklois/soft/ljc_methods/faceswap_gan/test_video/", "GPU_ID": 0, "Source": "test.mp4", "Model": "johnson2trump"}'