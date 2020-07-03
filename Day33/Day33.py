#!/usr/bin/env python
# coding: utf-8

# ## YOLO 細節理解 - 網路輸出的後處理    
# 今天的課程，我們說明了NMS在yolo中運作的情形:
# * NMS在YOLO的實際運作以每一個類別為主，各別執行NMS。     
# * YOLO 在NMS中採用的信心度為「每個 bbox 包含各類別的信心度」      
# 

# ### 作業
# 在NMS流程中，IoU重疊率參數(nms_threshold )調高，試著思考一下輸出的預測框會有甚麼變化?
# Hint: 可以回頭看今天介紹的內容，思考輸出的預測框變多或變少?
# 

# In[1]:


'''
輸出的預測筐會變多. 
考慮極端的情況, 如果nms_threshold = 1, 那麼只有完全和選定框(最高信心度之框)一樣的預測框才會被suppressed, 輸出將會近乎毫無篩選 
'''


