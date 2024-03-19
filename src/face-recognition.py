import cv2
import psycopg2
from PIL import Image
import pandas as pd
from imgbeddings import imgbeddings
from IPython.display import display

# 顔検出のアルゴリズムを指定
haar_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
# 内閣のリスト
cabinets = ['abe', 'fukuda', 'kishida', 'suga']

# 画像のリストを空の配列で
face_files = []

# 内閣の画像から顔を取得して保存
for name in cabinets:
    original_file_name = 'images/cabinet/' + name + '.jpg'
    img = cv2.imread(original_file_name, 0)
    # creating a black and white version of the image
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    faces = haar_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=10, minSize=(50, 50))
    i = 0
    for x, y, w, h in faces:
        # crop the image to select only the face
        cropped_image = img[y : y + h, x : x + w]
        # loading the target image path into target_file_name variable
        target_file_name = 'images/faces/' + name + str(i) + '.jpg'
        cv2.imwrite(
            target_file_name,
            cropped_image,
        )
        face_files.append({
            'file_name': target_file_name,
            'original_file_name': original_file_name
        })
        i += 1


ibed = imgbeddings()

# 画像のベクトルを取得してデータベースへ保存
conn = psycopg2.connect('dbname=app user=default password=secret host=db')
cur = conn.cursor()
for face_file in face_files:
    img = Image.open(face_file['file_name'])
    embedding = ibed.to_embeddings(img)[0]
    cur.execute(
        "INSERT INTO faces (original_file_name, file_name, embedding) VALUES (%s, %s, %s)",
        (face_file['original_file_name'], face_file['file_name'], embedding.tolist())
    )
    print (face_file['file_name'])
conn.commit()
conn.close()

# 検索したい顔画像からベクトルを取得
search_img = cv2.imread('images/shinzo.jpg', 0)
gray_search_img = cv2.cvtColor(search_img, cv2.COLOR_RGB2BGR)
search_faces = haar_cascade.detectMultiScale(gray_search_img, scaleFactor=1.1, minNeighbors=10, minSize=(50, 50))
(x, y, w, h) = search_faces[0]
cropped_image = search_img[y: y + h, x: x + w]
pil_img = Image.fromarray(cropped_image)
ibed = imgbeddings()
embedding = ibed.to_embeddings(pil_img)[0]

vector_string = "["
for x in embedding.tolist():
    vector_string += str(x) + ","
vector_string = vector_string[:-1] + "]"


conn = psycopg2.connect('dbname=app user=default password=secret host=db')
cur = conn.cursor()
cur.execute(
    "SELECT id, original_file_name, file_name , embedding <-> %s as distance, embedding <=> %s as cos_similarity, embedding <#> %s as inner_product FROM faces ORDER BY embedding <=> %s LIMIT 5;",
    (vector_string,vector_string,vector_string,vector_string)
)

dict_result = []
rows = cur.fetchall()
for row in rows:
    dict_result.append(dict(row))

for row in dict_result:
    img = Image.open(row['file_name'])
    display(img)

pd.DataFrame(dict_result)

for row in dict_result:
    img = Image.open(row['file_name'])
    display(img)


