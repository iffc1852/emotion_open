創作者使用的系統為Windows 11，不知道Linux系統等的安裝問題。

請下載FFmpeg並在環境變數裡面的Path加入他

請在python 3.10.11 環境下運作，在建立乾淨環境後使用pip install -r requirements-cpu.txt

要用uv的話，請用這個指令uv pip install -r requirements-cpu.txt

主程式為main.py

第一次運行會特別久，需要下載檔案。

whisper STT 設定為CPU方式


LLM為了方便而使用線上openai的chat-4o，私下則使用glm-4-9b-chat-GGUF，有明顯的差距。

有使用視訊鏡頭－麥克風－播放系統，請確定有正常運作或配置正常，如有問題，可以配合hardware\_setup.py來尋找並做更改。


大多數程式要在獨立GPU下運行才會有比較好的速度，CPU運行會非常慢。請注意！

可改為GPU方式的有whisper, py-feat.cosyvoice2內建使用GPU


#  							注意



#### 【硬體與系統要求】

#### 

#### NVIDIA GPU (VRAM > 6GB 尤佳)。

#### 

#### 必須手動安裝 NVIDIA CUDA Toolkit 12.1 版本。

#### 

#### 安裝完成後，請確認 CUDA 的 bin 資料夾已加入系統的 PATH 環境變數中。



具備以上請使用requirements-gpu.txt



不具備則使用requirements-cpu.txt


