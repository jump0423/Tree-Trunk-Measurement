#輔負責
class TrunkDetector:

 def __init__(self, model_path="best.pt"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到模型檔：{model_path}")
        print("正在載入 YOLO 模型...")
        self.model = YOLO(model_path)

 def detect_and_measure(self, image_path):
        """
        執行偵測，回傳一個 list，但現在保證 list 裡面最多只有 1 個結果 (信心度最高者)。
        """
        img = cv2.imread(image_path)
        if img is None:
            return None, []

        height, width, _ = img.shape
        
        # ---------------------------------------------------------
        # 信心度門檻 (conf) 為 0.4
        # 且保持 save=False (結果由 main 儲存)
        # ---------------------------------------------------------
        results = self.model.predict(
            source=image_path, 
            save=False,       
            verbose=False, 
            conf=0.4  # 低於 0.4 的不會被偵測出來
        )
        
        detected_trees = [] 

        for result in results:
            # 檢查是否有偵測到任何物體
            if result.masks and len(result.boxes) > 0:
                
                # 選取信心度最高的那一個
                # 取得所有框的信心度分數
                conf_scores = result.boxes.conf.cpu().numpy()
                
                # 找出最高分的索引 (index)
                best_idx = np.argmax(conf_scores)
                
                # 只取出最高分對應的那個遮罩
                mask_tensor = result.masks.data[best_idx]
                
                # --- 以下處理邏輯只會執行一次 ---
                
                # 1. 取得遮罩
                mask_data = mask_tensor.cpu().numpy()
                mask_resized = cv2.resize(mask_data, (width, height))
                
                # 2. 找出樹的垂直範圍 (Y軸)
                y_indices, x_indices = np.where(mask_resized > 0.5)
                
                if len(y_indices) > 0:
                    y_top = np.min(y_indices)
                    y_bottom = np.max(y_indices)
                    
                    # 3. 算出中間高度
                    measure_y = int((y_top + y_bottom) / 2)
                    
                    # 4. 在該高度找出左右邊緣 (X軸)
                    row_pixels = mask_resized[measure_y, :]
                    tree_indices = np.where(row_pixels > 0.5)[0]
                    
                    if len(tree_indices) > 0:
                        x_start = np.min(tree_indices)
                        x_end = np.max(tree_indices)
                        pixel_width = x_end - x_start
                        
                        tree_info = {
                            "mask": mask_resized,
                            "measure_y": measure_y,
                            "x_start": x_start,
                            "x_end": x_end,
                            "pixel_width": pixel_width,
                            "confidence": conf_scores[best_idx] # (選填) 也可以順便存信心度
                        }
                        detected_trees.append(tree_info)
        
        # 回傳的 detected_trees 裡面現在最多只會有一棵樹
        return img, detected_trees