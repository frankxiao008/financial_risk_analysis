# Task 1 赛题理解

根据贷款申请人的数据信息预测其是否有违约的可能，以此判断是否通过此项贷款，这是一个典型的分类问题。 
目标： 提交结果为每个测试样本是1的概率，也就是y为1的概率。评价方法为AUC评估模型效果（越大越好）

- 提交结果是概率，而不是针对测试集的预测值
- 评价方法是AUC

![formula for FPR, TPR](https://github.com/frankxiao008/financial_risk_analysis/blob/master/images/Finanlre.PNG)


### ROC 曲线

![AUC curve](https://github.com/frankxiao008/financial_risk_analysis/blob/master/images/330px-Roccurves.png)

The ROC curve is created by plotting the true positive rate (TPR) against the false positive rate (FPR) at various threshold settings
**TPR = TP/P =TP/(TP + FN)**  
**FPR = FP/N =FP/(FP + TN)**

数据集的字段结构

Field	Description
id	为贷款清单分配的唯一信用证标识                       
loanAmnt	贷款金额
term	贷款期限（year）  **整数型**
interestRate	贷款利率
installment	分期付款金额
grade	贷款等级                  **离散型**
subGrade	贷款等级之子级         **离散型**       
employmentTitle	就业职称        **离散型**
employmentLength	就业年限（年） **整数型**
homeOwnership	借款人在登记时提供的房屋所有权状况  **boolean**
annualIncome	年收入
verificationStatus	验证状态   **boolean**
issueDate	贷款发放的月份    **离散型**
purpose	借款人在贷款申请时的贷款用途类别  
postCode	借款人在贷款申请中提供的邮政编码的前3位数字
regionCode	地区编码
dti	债务收入比
delinquency_2years	借款人过去2年信用档案中逾期30天以上的违约事件数    
ficoRangeLow	借款人在贷款发放时的fico所属的下限范围
ficoRangeHigh	借款人在贷款发放时的fico所属的上限范围
openAcc	借款人信用档案中未结信用额度的数量
pubRec	贬损公共记录的数量
pubRecBankruptcies	公开记录清除的数量
revolBal	信贷周转余额合计
revolUtil	循环额度利用率，或借款人使用的相对于所有可用循环信贷的信贷金额
totalAcc	借款人信用档案中当前的信用额度总数
initialListStatus	贷款的初始列表状态
applicationType	表明贷款是个人申请还是与两个共同借款人的联合申请  
earliesCreditLine	借款人最早报告的信用额度开立的月份
title	借款人提供的贷款名称
policyCode	公开可用的策略_代码=1新产品不公开可用的策略_代码=2  **离散型**
n系列匿名特征	匿名特征n0-n14，为一些贷款人行为计数特征的处理    


https://blog.csdn.net/u012735708/article/details/86507026

ROC曲线及AUC系数主要用来检验模型对客户进行正确排序的能力。ROC曲线描述了在一定累计好客户比例下的累计坏客户的比例，模型的分别能力越强，ROC曲线越往左上角靠近。AUC系数表示ROC曲线下方的面积。AUC系数越高，模型的风险区分能力越强。
KS（Kolmogorov-Smirnov）检验:K－S检验主要是验证模型对违约对象的区分能力，通常是在模型预测全体样本的信用评分后，将全体样本按违约与非违约分为两部分，然后用KS统计量来检验这两组样本信用评分的分布是否有显著差异。

ROC值一般在0.5-1.0之间。值越大表示模型判断准确性越高，即越接近1越好。ROC=0.5表示模型的预测能力与随机结果没有差别。
KS值表示了模型将+和-区分开来的能力。值越大，模型的预测准确性越好。一般，KS>0.2即可认为模型有比较好的预测准确性。

要弄明白ks值和auc值的关系首先要弄懂roc曲线和ks曲线是怎么画出来的。其实从某个角度上来讲ROC曲线和KS曲线是一回事，只是横纵坐标的取法不同而已。拿逻辑回归举例，模型训练完成之后每个样本都会得到一个类概率值（注意是类似的类），把样本按这个类概率值排序后分成10等份，每一份单独计算它的真正率和假正率，然后计算累计概率值，用真正率和假正率的累计做为坐标画出来的就是ROC曲线，用10等分做为横坐标，用真正率和假正率的累计值分别做为纵坐标就得到两个曲线，这就是KS曲线。AUC值就是ROC曲线下放的面积值，而ks值就是ks曲线中两条曲线之间的最大间隔距离。由于ks值能找出模型中差异最大的一个分段，因此适合用于cut_off，像评分卡这种就很适合用ks值来评估。但是ks值只能反映出哪个分段是区分最大的，而不能总体反映出所有分段的效果，因果AUC值更能胜任。

K-S曲线与ROC曲线类似

- ROC曲线将真正例率和假正例率作为横纵轴
- K-S曲线将真正例率和假正例率都作为纵轴，横轴则由选定的阈值来充当。
  公式如下：
  $$KS=max(TPR-FPR)$$
  KS不同代表的不同情况，一般情况KS值越大，模型的区分能力越强，但是也不是越大模型效果就越好，如果KS过大，模型可能存在异常，所以当KS值过高可能需要检查模型是否过拟合。以下为KS值对应的模型情况，但此对应不是唯一的，只代表大致趋势。
- KS值<0.2,一般认为模型没有区分能力。
- KS值[0.2,0.3],模型具有一定区分能力，勉强可以接受
- KS值[0.3,0.5],模型具有较强的区分能力。
- KS值大于0.75，往往表示模型有异常。

