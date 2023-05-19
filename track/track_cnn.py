from ultralytics import YOLO
import cv2


video = cv2.VideoCapture('video_juste_nid-test.ts')
#video = cv2.VideoCapture('test_nid_vert.MTS')
#video = cv2.VideoCapture('video_limitee_nourriture-test.ts')
#video = cv2.VideoCapture('video_boite_entiere-test.ts')
model = YOLO('best_2.pt')

seuil = 0.1

classe = {
    0: 'f',
    1: 'r'
}


def zoom_at(img, zoom=1, angle=0, coord=None):
    
    cy, cx = [ i/2 for i in img.shape[:-1] ] if coord is None else coord[::-1]
    
    rot_mat = cv2.getRotationMatrix2D((cx,cy), angle, zoom)
    result = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
    
    return result

while True:
    _, frame = video.read()
    frame = zoom_at(frame,zoom=2.3,coord=(900,900))
    resultats = model(frame)[0]
    for res in resultats.boxes.data.tolist():
        x1, y1, x2, y2, score, id_classe = res
        if score > seuil:
            if int(id_classe) == 0:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            else:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 4)
            #cv2.putText(frame, classe[int(id_classe)].upper(), (int(x1), int(y1 - 10)),
                        #cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()