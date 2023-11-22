import os
from PIL import Image 
from sentence_transformers import SentenceTransformer 

def get_common_google_colab():  
    res = [
        "from google.colab import auth", 
        "auth.authenticate_user()", 
        
    ] 

    return res 

def get_image_name_list(images_dir):
    file_list = os.listdir(images_dir)
    res = []

    accepted_suffix = ("jpg", "png", "jpeg")
    for file in file_list:
        if file.lower().endswith(accepted_suffix):
            file = os.path.join(images_dir, file)
            res.append(file)
    return res



def load_image_list(image_name_list):
    img_list = []

    for file_name in image_name_list:
        # print(file_name)
        im = Image.open(file_name)
        img_list.append(im)

    return img_list


def get_model_from_gs(gs_path): 
    print("given the goodle storage path, return the model ") 
    temp_data_path = "temp_model.zip"  
    print(gs_path)
    os.system(f"gsutil cp {gs_path} ./{temp_data_path}")
    temp_unzip_model_path = "eval_model"
    os.system(f"unzip ./{temp_data_path} -d ./{temp_unzip_model_path}") 
    print("temp_unzip_model_path: ",temp_unzip_model_path)  

    target_model_path = "cur_model_path" 

    model_checkpoint_path_local = os.path.join(temp_unzip_model_path, target_model_path)    


    print("removing the previous folder: ", model_checkpoint_path_local) 
    os.system(f"rm -rf {model_checkpoint_path_local}") 

    print("rename the model file name to our sepcified. ")
    os.system(f"mv {temp_unzip_model_path}/* {model_checkpoint_path_local}") 

    print("removing the zip file ")
    os.system(f"rm -rf {temp_data_path}")


    model = SentenceTransformer(model_checkpoint_path_local)
    model.eval() 
    return model 


def get_evaluate_data_from_gs(gs_path, target_folder_path="data_path"): 
    print("given the gs path, return the evaluate dataset. ")  

    temp_data_path = "temp_data.zip"  
    os.system(f"gsutil cp {gs_path} ./{temp_data_path}")
    os.system(f"unzip ./{temp_data_path} -d ./{target_folder_path}") 
    target_subfolder_path="cur_data_path"
    data_path_local = os.path.join(target_folder_path, target_subfolder_path)    

    # print("removing the previous folder: ", data_path_local) 
    os.system(f"rm -rf {data_path_local}") 
    # print("rename the model file name to our sepcified. ")
    os.system(f"mv {target_folder_path}/* {data_path_local}") 
    # print("removing the zip file ")
    os.system(f"rm -rf {temp_data_path}")   

    image_name_list = get_image_name_list(data_path_local) 
    loaded_image_list = load_image_list(image_name_list) 

    return loaded_image_list

