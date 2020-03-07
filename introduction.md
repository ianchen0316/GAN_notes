# [筆記] 李宏毅 - GAN Lecture 1

### Introduction of Generative Adversarial Network

- 在了解GAN之前, 我們先來看看GAN中的"Generative"代表什麼。
- Generative Model可以將任何在範圍內的input vectors, 生成特定的資料型態。
- 譬如下圖, 我們可以從normal distribution中隨機的sample出一個vector, 丟進generative model中。根據model被train的型態, 會產生該vector對應的資料型態, 例如影像/文字/訊號 ...。

![](https://i.imgur.com/eI4Qc3f.png)

- 那這樣的隨機產生特定資料型態的model, 有甚麼用處呢? 其實用途是不大的。在往後我們會看到conditional GAN這樣的方法。在cGAN下, 我們可以去控制給定一個資料型態, 譬如一張圖片或一段話, cGAN model應該要生成什麼樣圖片或句子。這樣的方法, 就有很多潛在的應用空間。

- 但目前, 我們還是先來看看在給定隨機向量底下, generative model要如何生成想要的資料型態吧!

-----------------------------------------------------------

- 以圖像生成的例子來說, generative model所做的事其實就是把一個input向量打到一個很高維度、代表圖像pixel的向量。

- 而通常，input向量中的每個維度都代表著某些"特徵"。例如下圖的例子, input向量的第一個維度可能代表著"頭髮長度", 最後一個維度可能代表"嘴巴大小"。而generative model就會根據這樣的指定特徵向量, 去生成具有這些特徵的對應圖像。

![](https://i.imgur.com/p3jA3z8.png)

-----------------------------------------------------------

- 那這樣的generative model, 要怎麼訓練出來呢? 
- 這就是GAN這個架構很有代表性的地方。GAN採用了"generator"和"discriminator"交互訓練的機制。剛剛已經介紹generator是將input vector生成我們想要的資料型態。而discriminator的任務就是去檢查剛剛generator所生成的資料型態, 夠不夠"擬真"。
- 因此, discriminator的input就會是想生成的資料型態, 而output會是一個scaler,代表著input的"擬真程度"。
- 以影像生成來說, discriminator的input就會是一張影像, 而output會是該影像作為動漫圖片的擬真指數。


![](https://i.imgur.com/F311Tsb.png)

------------------------------------------------------------

- 因此, generator和discriminator彼此就像是獵食者與被獵食者的演化關係一般。
- Generator會為了產生讓discriminator辨識不出的output, 而努力演化。Discriminator也會為了分辨generator產生之output和真實output之不同而努力演化。

![](https://i.imgur.com/j3OaZcY.png)

- 放到圖像生成的例子來說, 一開始generator_v1什麼都不會, 因此會產生很糟的圖片。而discriminator_v1能輕易將這些很糟的圖片與真實圖片做區分。但接下來, generator_v2也會以騙過discriminator_v1為目標, 生成更高品質的圖片。作為對應, discriminator也會變得更強, 去區分進化後的圖片與真實圖片的差異。

- 因此, 這樣的過程是"對抗的"。故有GAN中, "adversarial"這樣的稱呼。

![](https://i.imgur.com/GcEXjWO.png)

-----------------------------------------------------------------

- 從演算法來看GAN的話, 分為幾個步驟:

    - 首先, 我們會先隨機初始化generator和discriminator兩個networks
    - 接下來, 我們會先固定generator而去訓練discriminator。透過隨機產生一組向量, 丟給generator去生成想要的資料型態。而discriminator會被訓練去盡量分辨由generator所生成的object(label=0), 以及真實資料(label=1)
    - 再來, 我們會固定discriminator並訓練generator。透過隨機把向量丟給generator, 使得generator產生object, 再將object丟給discriminator去給出一個分數, 目標是使得分數越高越好。**而注意這個過程中, 需要固定discriminator的參數, 只調generator的參數。**

![](https://i.imgur.com/lI0vq1U.png)

![](https://i.imgur.com/NZylK5W.png)

----------------------------------------

- 以下是基本的GAN在演算法上的總結:
- 注意到此處在訓練generator和discriminator時,都使用了cross-entropy loss。這是原始論文的作法,但並非是目前最好的作法。往後我們會看到演進的地方。

![](https://i.imgur.com/uIl1TbY.png)

----------------------------------------

- 接下來, 我們要探討, GAN是否可以作為structured learning的一種方法。
- 回顧一下, 與regression和classification問題不同, structured learning的output會是存在著dependency的資料結構, 譬如序列、矩陣、graph、tree等等。

![](https://i.imgur.com/FLOdH45.png)

- 對於翻譯、語音辨識、聊天機器人來說, output是個序列, 因此屬於structured learning的問題。 

![](https://i.imgur.com/NYjsWMB.png)

- 對於影像轉換、文字轉影像來說, output會是一個矩陣, 因此也屬於structured learning的範疇。

![](https://i.imgur.com/lRe4SPB.png)

------------------------------------------------------------

- 那為什麼structured learning會是滿有挑戰性的問題呢? 
- 在classification中, 通常我們每個類別都會有不少訓練資料。相對之下, 在structured learning中, 如果我們把每個可能的output視為一種類別, 那大多數類別是只有非常少量,甚至是沒有對應的訓練資料的。
- 因此, 從這個角度出發, structured learning更像是few-shot, 甚至是zero-shot learning。
- 機器本身需要有"創造"的能力,才能產生好結果。

![](https://i.imgur.com/PEbRqBE.png)

- 另外, 在structured learning中, 演算法要做的好, 還需要有"大局觀"。由於output是具有dependency的資料結構, 還必須兼顧output中的某一個點與其他點之間的關聯性有沒有正確的被產生。
- 例如下圖中, 圖片中間產生一個黑點這件事本身是中性的。但如果其他點最後是連成一條線, 那就是一個好的數字output。相反的, 如果其他點最後連成一個圓, 那就不是好的數字output。

![](https://i.imgur.com/yYnnLcd.png)

--------------------------------------------------------

- 一般來說, 在structured learning中, 有兩大類方法:

    - 一種是bottom-up方法，著重在如何良好的生成一個個資料結構中的component,使其整體組合起來為好的output。但缺點就是大局觀難以抓到。
    - 另一種方式是top-down, 著重在保留大局觀以作為生成好的output的原則。但缺點就是細節難以做的好。
- GAN則是結合了兩種方法。Generator代表著bottom-up方式, 而Discriminator代表著top-down方式。 

![](https://i.imgur.com/b2kGZzW.png)

-----------------------------------------------

- 剛剛看到了GAN是bottom-up和top-down兩種方式的結合。作為比較，便又回到了原先的問題，能不能只透過generator,就生成好的output,把structured learning做好呢?
- 讓我們來嘗試訓練一個NN Generator去做圖像生成看看:

    - 首先由於NN Generator的input是各種向量，我們會給每張訓練圖片指定一個對應的向量作為input，而output則是該圖片所拉長而成的高維度向量。
    - 比較一下，圖像分類中的input是圖片,而output是各分類機率向量。但訓練generator做圖像生成時, input是某種encoder向量, 而output是圖片對應的高維度向量。
    - 然而, 訓練network的機制是完全相同的! 因此,單純使用generator去做圖像生成,理應當是沒有問題的。

- 剩下的問題就是, 一開始給予訓練圖片的對應encode input向量, 要如何給定? 我們會希望在相似的圖片, 例如各種1的型態, 對應的encode向量也會彼此比1和2的型態更為接近。因此隨機的去指定encode向量是不太可行的。

![](https://i.imgur.com/Wut3NND.png)

- 我們可以使用autoencoder去做到這樣的encoding。透過encoder和decoder的自我訓練, 我們就可以從encoder中產生能丟給generator的input。

![](https://i.imgur.com/vDkRFdk.png)

![](https://i.imgur.com/536pATQ.png)

- 而為了使autoencoder更穩定, 我們還可以其中encoder產生的結果加入雜訊, 使得整體network比較能學到穩健的encoder vector。

![](https://i.imgur.com/KRRnBu6.png)
