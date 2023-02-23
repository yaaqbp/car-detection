from torch.hub import load as torch_load
from PIL import Image
import matplotlib.pyplot as plt
import io
import pickle

class YOLO():
    def __init__(self):
        self.yolo = torch_load('ultralytics/yolov5', 'yolov5s')
        #self.yolo = pickle.load(open('yolo.pickle', 'rb'))

    def find_cars(self, image_bytes):
        image = Image.open(io.BytesIO(image_bytes))
        result = self.yolo(image)
        df_result = result.pandas().xyxy[0]
        df_result = df_result[((df_result.name == 'car')|(df_result.name == 'bus')|(df_result.name == 'truck'))&(df_result.confidence > 0.22)]
        
        fig = plt.figure(figsize=(5,5))
        plt.imshow(image)
        plt.axis('off')
        for row in df_result.iterrows():
            xmin = row[1].xmin
            ymin = row[1].ymin
            xmax = row[1].xmax
            ymax = row[1].ymax
            plt.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], 'r-')
        plt.subplots_adjust(0, 0, 1, 1)
        imgdata = io.StringIO()
        fig.savefig(imgdata, format='svg', transparent=True)
        imgdata.seek(0)
        data = imgdata.getvalue()

        return df_result.shape[0], data
        