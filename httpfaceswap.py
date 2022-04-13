from fastapi import FastAPI,File, UploadFile,Form,Response
from fastapi.responses import HTMLResponse,FileResponse

import requests
import os
import time

import shutil
app = FastAPI()

@app.post('/api')
async def faceswap(mtype:str = Form(...), file:UploadFile = File(...)):
    if os.path.exists(r".\test_result"):
        shutil.rmtree(r".\test_result")
    os.mkdir(r".\test_result")
    if os.path.exists(r".\test_video"):
        shutil.rmtree(r".\test_video")
    os.mkdir(r".\test_video")
    contents = await file.read()
    with open(r".\test_video\test.mp4", "wb") as f:
        f.write(contents)
    #action = request_data.mtype
    if mtype =='obama2biden':
        os.system(r"python FaceSwap_GAN_v2.2_video_conversion.py --input .\test_video\test.mp4 --output .\test_result\result.mp4 --face-size 64 -g 0 --model-dir .\model\model_3_4 --merge-type BtoA --mode video")
    elif mtype =='biden2obama':
        os.system(r"python FaceSwap_GAN_v2.2_video_conversion.py --input .\test_video\test.mp4 --output .\test_result\result.mp4 --face-size 64 -g 0 --model-dir .\model\model_3_4 --merge-type AtoB --mode video")
    elif mtype == 'johnson2trump':
        os.system(r"python FaceSwap_GAN_v2.2_video_conversion.py --input .\test_video\test.mp4 --output .\test_result\result.mp4 --face-size 64 -g 0 --model-dir .\model\model_1_2 --merge-type AtoB --mode video")
    elif mtype == 'trump2johnson':
        os.system(r"python FaceSwap_GAN_v2.2_video_conversion.py --input .\test_video\test.mp4 --output .\test_result\result.mp4 --face-size 64 -g 0 --model-dir .\model\model_1_2 --merge-type BtoA --mode video")
    
    return FileResponse(r".\test_result\result.mp4")

@app.get('/api')
def get_video():
    return FileResponse(r".\test_result\result.mp4")

    
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app=app,
                host="127.0.0.1",
                port=7071,
                workers=1)