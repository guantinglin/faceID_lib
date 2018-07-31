# faceID_lib

# How to run
If you want to find the faces in the picture
cd $faceID_root
python3 examples/find_face_in_picture/find_faces_in_picture.py --input image_to_test --output output_image_with_face_rect

Search specific IDs with sample image in images folder
cd $faceID_root
python3 examples/search_ID/search.py --sample examples/search_ID/sample_clintion_trump.png --input examples/search_ID/input/ --output_folder examples/search_ID/output/ --cpus 1 --model cnn
The output will be save in $faceID_root/examples/search_ID/output

