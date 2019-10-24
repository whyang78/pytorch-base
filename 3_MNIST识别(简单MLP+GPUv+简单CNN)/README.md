1、torch中主要操作tensor类型的数据，注意其包含数据和梯度信息，所以   
 若要单独取出数据，可以用item或者data取出。前者取出的是float类型,后者是tensor data类型.   
 例如：网络中算出来的loss，想要累加要取出数值，则用loss.item()或loss.data
 
2、在使用cuda时，注意以下几个问题：   
（1）需要先指定cuda，网络、损失函数、输入需要转移到cuda上使用   
（2）在cuda上运算之后的结果类型就变成了tensor cuda类型，所以原始标签需要转换类型为cuda。   
（3）注意loss累加时，不要累加梯度，否则显存会很快爆满。一定要注意取loss.data 或 loss.item()
