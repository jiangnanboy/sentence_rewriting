## chinese_sentence_rewriting

pytorch code for chinese sentence rewriting.

chinese blog [gpt2句子改写生成](https://www.cnblogs.com/little-horse/p/15795468.html).

一.说明

    参考我之前的一个项目给定title和keywords利用gpt2生成文本，从中可以看出只是利用了gpt2模型，没有其它模型上的改动或组合，变化的只是input部分，在input中，加入了title和keywords两部分。那么训练时候的model输入，就会有三部分：[BOS] + title + [SEP] + keywords + [SEP] + text + [EOS]，所以生成的文章会与title和keywords有关。

    根据以上，我们可以做一个小的实验，就是针对一个句子进行改写生成，或者説是可控生成，就是生成的句子不能偏离原句的意思。此实验是基于以上项目的改动，改动的部分只是input部分，这里只需去除title，针对句子提取keywords，输入：[BOS] + keywords + [SEP] + text + [EOS]，这样生成的时候不会偏离原句大意。

二.src/sentence_rewriting/:利用pytorch进行训练和推理

    python train.py
    python generate.py
	
三.src/convert/:将pytorch模型转为onnx格式进行推理
	python model_convert.py
	python onnx_generate.py

四.实验结果

    从结果中可看到有那么一点意思，不过整体还不够准确，句子改动过大。后期会考虑加入同义词这种外部知识来进行优化。

![image](https://raw.githubusercontent.com/jiangnanboy/sentence_rewriting/master/image/result.png)

## Reference
- https://github.com/jiangnanboy/text_generation

