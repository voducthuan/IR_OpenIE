# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 07:43:54 2021

@author: Vo Duc Thuan
"""
import matchzoo as mz
import pandas as pd

#df_train = pd.read_csv("C:/matchzoo/clueweb_clean/sample_data/train_small.csv")
#df_test = pd.read_csv("C:/matchzoo/clueweb_clean/sample_data/test_small.csv")
#df_train = pd.read_csv("/home/thuan/clueweb/clean/2_query/final_topic_query_2(query)_train.csv")
#df_test = pd.read_csv("/home/thuan/clueweb/clean/2_query/final_topic_query_2(query)_test.csv")
#df_train = pd.read_csv("/home/thuan/clueweb/clean/2_des/final_topic_query_2(des)_train.csv")
#df_test = pd.read_csv("/home/thuan/clueweb/clean/2_des/final_topic_query_2(des)_test.csv")
#original option 1
#df_train = pd.read_csv("/home/thuan/clueweb/org/1_query/final_org_1_topicCombine_train(query)_train.csv")
#df_test = pd.read_csv("/home/thuan/clueweb/org/1_query/final_org_1_topicCombine_test(query)_test.csv")
#original option 2
#df_train = pd.read_csv("/home/thuan/clueweb/org/2_query/final_org_2_TopicCombine_train(query)_train.csv")
#df_test = pd.read_csv("/home/thuan/clueweb/org/2_query/final_org_2_TopicCombine_test(query)_test.csv")

#df_train = pd.read_csv("/home/thuan/clueweb/clean/org/1_des/final_org_1_topicCombine_train(des)_train.csv")
#df_test = pd.read_csv("/home/thuan/clueweb/clean/org/1_des/final_org_1_topicCombine_test(des)_test.csv")
#original option 2
#df_train = pd.read_csv("/home/thuan/clueweb/clean/org/2_des/final_org_2_TopicCombine_train(des)_train.csv")
#df_test = pd.read_csv("/home/thuan/clueweb/clean/org/2_des/final_org_2_TopicCombine_test(des)_test.csv")

#Robust data
#Orginal Option 2
#df_train = pd.read_csv("/home/thuan/robust/final/org/2_query/final_robust_topCom1(query)_train.csv")
#df_test = pd.read_csv("/home/thuan/robust/final/org/2_query/final_robust_topCom1(query)_test.csv")

#option oie
#df_train = pd.read_csv("/home/thuan/robust/final/2_query/final_robust_topCom(query)_train_new.csv")
#df_test = pd.read_csv("/home/thuan/robust/final/2_query/final_robust_topCom(query)_test.csv")
#df_train = pd.read_csv("/home/thuan/robust/final/v3_query/final_robust_v3_topCom(query)_train.csv")
#df_test = pd.read_csv("/home/thuan/robust/final/v3_query/final_robust_v3_topCom(query)_test.csv")

#Clueweb 12 data
#df_train = pd.read_csv("/home/thuan/clueweb12/final/org/1_query/final_org1_clueweb12(query)_topCom_train.csv")
#df_test = pd.read_csv("/home/thuan/clueweb12/final/org/1_query/final_org1_clueweb12(query)_topCom_test.csv")
#df_train = pd.read_csv("/home/thuan/clueweb12/final/1_query/final_clueweb12_topCom(query)_train.csv")
#df_test = pd.read_csv("/home/thuan/clueweb12/final/1_query/final_clueweb12_topCom(query)_test.csv")

#Gov2 data
#df_train = pd.read_csv("/home/thuan/clueweb12/final/1_query/final_clueweb12_topCom(query)_train.csv")
#df_test = pd.read_csv("/home/thuan/clueweb12/final/1_query/final_clueweb12_topCom(query)_test.csv")
df_train = pd.read_csv("/home/thuan/gov2/final/org/2_query/final_org2_gov(query)_train.csv")
df_test = pd.read_csv("/home/thuan/gov2/final/org/2_query/final_org2_gov(query)_test.csv")

ranking_task = mz.tasks.Ranking()
ranking_task.metrics = [
                 mz.metrics.MeanReciprocalRank(),
                 mz.metrics.NormalizedDiscountedCumulativeGain(k=10),
                 mz.metrics.NormalizedDiscountedCumulativeGain(k=20),
                 mz.metrics.MeanAveragePrecision(),
                 mz.metrics.Precision(k=10),
                 mz.metrics.Precision(k=20)
             ]

train_raw = mz.pack(df_train)
test_raw = mz.pack(df_test)

type(train_raw)
train_raw.left.head()
train_raw.right.head()
train_raw.relation.head()
train_raw.frame().head()

#preprocessor = mz.preprocessors.BasicPreprocessor(fixed_length_left=30, fixed_length_right=200, remove_stop_words=False)
preprocessor = mz.preprocessors.BasicPreprocessor(fixed_length_left=20, fixed_length_right=500, remove_stop_words=False)
train_pack_processed = preprocessor.fit_transform(train_raw)
test_pack_processed = preprocessor.transform(test_raw)

glove_embedding = mz.datasets.embeddings.load_glove_embedding(dimension=50)

preprocessor.context

model = mz.models.MVLSTM()
model.params.update(preprocessor.context)
model.params['task'] = ranking_task
model.params['embedding_output_dim'] = 50
model.params['lstm_units'] = 50
model.params['top_k'] = 20
model.params['mlp_num_layers'] = 2
model.params['mlp_num_units'] = 10
model.params['mlp_num_fan_out'] = 5
model.params['mlp_activation_func'] = 'relu'
model.params['dropout_rate'] = 0.5
model.params['optimizer'] = 'adadelta'
model.guess_and_fill_missing_params()
model.build()
model.compile()
model.backend.summary()

embedding_matrix = glove_embedding.build_matrix(preprocessor.context['vocab_unit'].state['term_index'])
model.load_embedding_matrix(embedding_matrix)

pred_x, pred_y = test_pack_processed.unpack()
evaluate = mz.callbacks.EvaluateAllMetrics(model, x=pred_x, y=pred_y, batch_size=len(pred_y))

train_generator = mz.DataGenerator(
    train_pack_processed,
    batch_size=64
)
print('num batches:', len(train_generator))

history = model.fit_generator(train_generator, epochs=15, callbacks=[evaluate], workers=20, use_multiprocessing=True)
print('end')