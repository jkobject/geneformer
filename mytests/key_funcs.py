


def preprocess_classifier_batch(cell_batch, max_len):
    if max_len == None:
        max_len = max([len(i) for i in cell_batch["input_ids"]])
    def pad_label_example(example):
        example["labels"] = np.pad(example["labels"], 
                                   (0, max_len-len(example["input_ids"])), 
                                   mode='constant', constant_values=-100)
        example["input_ids"] = np.pad(example["input_ids"], 
                                      (0, max_len-len(example["input_ids"])), 
                                      mode='constant', constant_values=token_dictionary.get("<pad>"))
        example["attention_mask"] = (example["input_ids"] != token_dictionary.get("<pad>")).astype(int)
        return example
    padded_batch = cell_batch.map(pad_label_example)
    return padded_batch

# forward batch size is batch size for model inference (e.g. 200)
def classifier_predict(model, evalset, forward_batch_size, mean_fpr):
    predict_logits = []
    predict_labels = []
    model.eval()
    
    # ensure there is at least 2 examples in each batch to avoid incorrect tensor dims
    evalset_len = len(evalset)
    max_divisible = find_largest_div(evalset_len, forward_batch_size)
    if len(evalset) - max_divisible == 1:
        evalset_len = max_divisible
    
    max_evalset_len = max(evalset.select([i for i in range(evalset_len)])["length"])
    
    for i in range(0, evalset_len, forward_batch_size):
        max_range = min(i+forward_batch_size, evalset_len)
        batch_evalset = evalset.select([i for i in range(i, max_range)])
        padded_batch = preprocess_classifier_batch(batch_evalset, max_evalset_len)
        padded_batch.set_format(type="torch")
        
        input_data_batch = padded_batch["input_ids"]
        attn_msk_batch = padded_batch["attention_mask"]
        label_batch = padded_batch["labels"]
        with torch.no_grad():
            outputs = model(
                input_ids = input_data_batch.to("cpu"), 
                attention_mask = attn_msk_batch.to("cpu"), 
                labels = label_batch.to("cpu"), 
            )
            predict_logits += [torch.squeeze(outputs.logits.to("cpu"))]
            predict_labels += [torch.squeeze(label_batch.to("cpu"))]
            
    logits_by_cell = torch.cat(predict_logits)
    all_logits = logits_by_cell.reshape(-1, logits_by_cell.shape[2])
    labels_by_cell = torch.cat(predict_labels)
    all_labels = torch.flatten(labels_by_cell)
    logit_label_paired = [item for item in list(zip(all_logits.tolist(), all_labels.tolist())) if item[1]!=-100]
    y_pred = [vote(item[0]) for item in logit_label_paired]
    y_true = [item[1] for item in logit_label_paired]
    logits_list = [item[0] for item in logit_label_paired]
    # probability of class 1
    y_score = [py_softmax(item)[1] for item in logits_list]
    conf_mat = confusion_matrix(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    # plot roc_curve for this split
    plt.plot(fpr, tpr)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.show()
    # interpolate to graph
    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    return fpr, tpr, interp_tpr, conf_mat 

def vote(logit_pair):
    a, b = logit_pair
    if a > b:
        return 0
    elif b > a:
        return 1
    elif a == b:
        return "tie"
    
def py_softmax(vector):
	e = np.exp(vector)
	return e / e.sum()
    
# get cross-validated mean and sd metrics
def get_cross_valid_metrics(all_tpr, all_roc_auc, all_tpr_wt):
    wts = [count/sum(all_tpr_wt) for count in all_tpr_wt]
    print(wts)
    all_weighted_tpr = [a*b for a,b in zip(all_tpr, wts)]
    mean_tpr = np.sum(all_weighted_tpr, axis=0)
    mean_tpr[-1] = 1.0
    all_weighted_roc_auc = [a*b for a,b in zip(all_roc_auc, wts)]
    roc_auc = np.sum(all_weighted_roc_auc)
    roc_auc_sd = math.sqrt(np.average((all_roc_auc-roc_auc)**2, weights=wts))
    return mean_tpr, roc_auc, roc_auc_sd

# Function to find the largest number smaller
# than or equal to N that is divisible by k
def find_largest_div(N, K):
    rem = N % K
    if(rem == 0):
        return N
    else:
        return N - rem



# cross-validate gene classifier
def cross_validate(data, targets, labels, nsplits, subsample_size, training_args, freeze_layers, output_dir, num_proc):
    # check if output directory already written to
    # ensure not overwriting previously saved model
    model_dir_test = os.path.join(output_dir, "ksplit0/models/pytorch_model.bin")
    if os.path.isfile(model_dir_test) == True:
        raise Exception("Model already saved to this directory.")
    
    # initiate eval metrics to return
    num_classes = len(set(labels))
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []
    all_roc_auc = []
    all_tpr_wt = []
    label_dicts = []
    confusion = np.zeros((num_classes,num_classes))
    
    # set up cross-validation splits
    skf = StratifiedKFold(n_splits=nsplits, random_state=0, shuffle=True)
    # train and evaluate
    iteration_num = 0
    for train_index, eval_index in tqdm(skf.split(targets, labels)):
        if len(labels) > 500:
            print("early stopping activated due to large # of training examples")
            nsplits = 3
            if iteration_num == 3:
                break
        print(f"****** Crossval split: {iteration_num}/{nsplits-1} ******\n")
        # generate cross-validation splits
        targets_train, targets_eval = targets[train_index], targets[eval_index]
        labels_train, labels_eval = labels[train_index], labels[eval_index]
        label_dict_train = dict(zip(targets_train, labels_train))
        label_dict_eval = dict(zip(targets_eval, labels_eval))
        label_dicts += (iteration_num, targets_train, targets_eval, labels_train, labels_eval)
        
        # function to filter by whether contains train or eval labels
        def if_contains_train_label(example):
            a = label_dict_train.keys()
            b = example['input_ids']
            return not set(a).isdisjoint(b)

        def if_contains_eval_label(example):
            a = label_dict_eval.keys()
            b = example['input_ids']
            return not set(a).isdisjoint(b)
        
        # filter dataset for examples containing classes for this split
        print(f"Filtering training data")
        trainset = data.filter(if_contains_train_label, num_proc=num_proc)
        print(f"Filtered {round((1-len(trainset)/len(data))*100)}%; {len(trainset)} remain\n")
        print(f"Filtering evalation data")
        evalset = data.filter(if_contains_eval_label, num_proc=num_proc)
        print(f"Filtered {round((1-len(evalset)/len(data))*100)}%; {len(evalset)} remain\n")

        # minimize to smaller training sample
        training_size = min(subsample_size, len(trainset))
        trainset_min = trainset.select([i for i in range(training_size)])
        eval_size = min(training_size, len(evalset))
        half_training_size = round(eval_size/2)
        evalset_train_min = evalset.select([i for i in range(half_training_size)])
        evalset_oos_min = evalset.select([i for i in range(half_training_size, eval_size)])
        
        # label conversion functions
        def generate_train_labels(example):
            example["labels"] = [label_dict_train.get(token_id, -100) for token_id in example["input_ids"]]
            return example

        def generate_eval_labels(example):
            example["labels"] = [label_dict_eval.get(token_id, -100) for token_id in example["input_ids"]]
            return example
        
        # label datasets 
        print(f"Labeling training data")
        trainset_labeled = trainset_min.map(generate_train_labels)
        print(f"Labeling evaluation data")
        evalset_train_labeled = evalset_train_min.map(generate_eval_labels)
        print(f"Labeling evaluation OOS data")
        evalset_oos_labeled = evalset_oos_min.map(generate_eval_labels)
        
        # create output directories
        ksplit_output_dir = os.path.join(output_dir, f"ksplit{iteration_num}")
        ksplit_model_dir = os.path.join(ksplit_output_dir, "models/") 
        
        # ensure not overwriting previously saved model
        model_output_file = os.path.join(ksplit_model_dir, "pytorch_model.bin")
        if os.path.isfile(model_output_file) == True:
            raise Exception("Model already saved to this directory.")

        # make training and model output directories
        subprocess.call(f'mkdir {ksplit_output_dir}', shell=True)
        subprocess.call(f'mkdir {ksplit_model_dir}', shell=True)
        
        # load model
        model = BertForTokenClassification.from_pretrained(
            "../geneformer-12L-30M/",
            num_labels=2,
            output_attentions = False,
            output_hidden_states = False
        )
        if freeze_layers is not None:
            modules_to_freeze = model.bert.encoder.layer[:freeze_layers]
            for module in modules_to_freeze:
                for param in module.parameters():
                    param.requires_grad = False
                
        model = model.to(torch.device("cpu"))
        
        # add output directory to training args and initiate
        training_args["output_dir"] = ksplit_output_dir
        training_args_init = TrainingArguments(**training_args)
        
        # create the trainer
        trainer = Trainer(
            model=model,
            args=training_args_init,
            data_collator=DataCollatorForGeneClassification(),
            train_dataset=trainset_labeled,
            eval_dataset=evalset_train_labeled
        )

        # train the gene classifier
        trainer.train()
        
        # save model
        trainer.save_model(ksplit_model_dir)
        
        # evaluate model
        fpr, tpr, interp_tpr, conf_mat = classifier_predict(trainer.model, evalset_oos_labeled, 200, mean_fpr)
        
        # append to tpr and roc lists
        confusion = confusion + conf_mat
        all_tpr.append(interp_tpr)
        all_roc_auc.append(auc(fpr, tpr))
        # append number of eval examples by which to weight tpr in averaged graphs
        all_tpr_wt.append(len(tpr))
        
        iteration_num = iteration_num + 1
        
    # get overall metrics for cross-validation
    mean_tpr, roc_auc, roc_auc_sd = get_cross_valid_metrics(all_tpr, all_roc_auc, all_tpr_wt)
    return all_roc_auc, roc_auc, roc_auc_sd, mean_fpr, mean_tpr, confusion, label_dicts