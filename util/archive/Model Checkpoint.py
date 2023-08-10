from segment_anything import SamPredictor, sam_model_registry
sam = sam_model_registry["vit_h"](checkpoint="C:\AI\Chinese Academy of Science\SAM materials\sam_vit_h_4b8939.pth")
predictor = SamPredictor(sam)
predictor.set_image("C:\AI\Chinese Academy of Science\SAM materials\randomImage.jpg")
masks, _, _ = predictor.predict("black dog")
