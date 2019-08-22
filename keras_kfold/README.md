## kfold集合keras的使用

`kfold`用于交叉验证时使用，将样本分成多份，取其中一份作为验证集。

使用`StratifiedKFold`容易出现下面的错误：
`ValueError: Supported target types are: ('binary', 'multiclass'). Got 'continuous' instead.`

这时只要改成使用KFold就可以了 
