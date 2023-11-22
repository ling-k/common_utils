import matplotlib.pyplot as plt
from sklearn.decomposition import PCA 

def plot_pca_embedding(embeddings1, embeddings2, title="2D PCA", is_save=False):  
  plt.style.use("ggplot") 
  pca = PCA(n_components=2)

  embeddings_2d_1 = pca.fit_transform(embeddings1)
  embeddings_2d_2 = pca.fit_transform(embeddings2)

  embeddings_2d_1 = pca.transform(embeddings1)
  embeddings_2d_2 = pca.transform(embeddings2) 
  label_1 = "Acura Image" 
  label_2 = "Honda Image"
  fig, ax = plt.subplots()  
  ax.scatter(embeddings_2d_1[:, 0], embeddings_2d_1[:, 1], color='red', label=label_1)
  ax.scatter(embeddings_2d_2[:, 0], embeddings_2d_2[:, 1], color='blue', label=label_2)

  ax.legend()
  plt.title(title)

  if is_save:
    file_name = title + ".pdf"
    plt.savefig(file_name,  dpi=600)  

  plt.show()
