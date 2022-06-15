import os
from paddle import inference
from paddlenlp.data import Pad, Dict
import numpy as np
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.abspath(os.path.join(__dir__, '../../../')))
from tydi_canine_model import CanineTokenizerforQA


test_article = "The Nobel Prize in Literature (Swedish: Nobelpriset i litteratur) is awarded annually by the Swedish " \
               "Academy to authors for outstanding contributions in the field of literature. It is one of the five " \
               "Nobel Prizes established by the 1895 will of Alfred Nobel, which are awarded for outstanding " \
               "contributions in chemistry, physics, literature, peace, and physiology or medicine.[1] As dictated by " \
               "Nobel's will, the award is administered by the Nobel Foundation and awarded by a committee that " \
               "consists of five members elected by the Swedish Academy.[2] The first Nobel Prize in Literature was " \
               "awarded in 1901 to Sully Prudhomme of France.[3] Each recipient receives a medal, a diploma and a " \
               "monetary award prize that has varied throughout the years.[4] In 1901, Prudhomme received 150," \
               "782 SEK, which is equivalent to 8,823,637.78 SEK in January 2018."

test_question = "Who was the first Nobel prize winner for Literature?"


class InferenceEngine(object):
    def __init__(self, args):
        super(InferenceEngine, self).__init__()
        self.args = args

        # init inference_python engine
        self.predictor, self.config, self.input_tensors, self.output_tensors = self.load_predictor(
            os.path.join(args.model_dir, "inference_python.pdmodel"),
            os.path.join(args.model_dir, "inference_python.pdiparams"))

        # build transforms
        self.tokenizer = CanineTokenizerforQA()
        self.batchify_fn = Dict({
            "input_ids": Pad(axis=0, pad_val=self.tokenizer.pad_token_id, dtype="int64"),
            "token_type_ids": Pad(axis=0, pad_val=self.tokenizer.pad_token_id, dtype="int64"),
            "attention_mask": Pad(axis=0, pad_val=self.tokenizer.pad_token_id, dtype="int64")
        }
        )

        # wamrup
        if self.args.warmup > 0:
            for idx in range(args.warmup):
                input_ids = np.random.randint(
                    5, 200, size=(4, 32)).astype("int64")
                token_type_ids = np.ones_like(input_ids).astype(
                    "int64")
                attention_mask = np.ones_like(input_ids).astype(
                    "int64")

                self.input_tensors[0].copy_from_cpu(input_ids)
                self.input_tensors[1].copy_from_cpu(token_type_ids)
                self.input_tensors[2].copy_from_cpu(attention_mask)
                self.predictor.run()
                self.output_tensors[0].copy_to_cpu()
                self.output_tensors[1].copy_to_cpu()

        return

    def load_predictor(self, model_file_path, params_file_path):
        """load_predictor
        initialize the inference_python engine
        Args:
            model_file_path: inference_python model path (*.pdmodel)
            model_file_path: inference_python parmaeter path (*.pdiparams)
        Return:
            predictor: Predictor created using Paddle Inference.
            config: Configuration of the predictor.
            input_tensor: Input tensor of the predictor.
            output_tensor: Output tensor of the predictor.
        """
        args = self.args
        config = inference.Config(model_file_path, params_file_path)
        if args.use_gpu:
            # config.enable_use_gpu(1000, 0)
            pass
        else:
            config.disable_gpu()
            # The thread num should not be greater than the number of cores in the CPU.
            config.set_cpu_math_library_num_threads(4)  # TODO check threads

        config.enable_memory_optim()
        config.disable_glog_info()

        config.switch_use_feed_fetch_ops(False)
        config.switch_ir_optim(True)

        # create predictor
        predictor = inference.create_predictor(config)

        # get input and output tensor property
        input_tensors = [
            predictor.get_input_handle(name)
            for name in predictor.get_input_names()
        ]
        output_tensors = [
            predictor.get_output_handle(name)
            for name in predictor.get_output_names()
        ]

        return predictor, config, input_tensors, output_tensors

    def preprocess(self, article, question):
        """preprocess
        Preprocess to the input.
        Args:
            article: the context of article.
            question: the question.
        Returns: Input data after preprocess.
        """
        inputs = self.tokenizer([[question, article]],
                                return_tensors='np',
                                return_attention_mask=True,
                                return_token_type_ids=True, )
        # TODO, only supports 1 batch size now
        # inputs = self.batchify_fn(inputs)

        return inputs

    def postprocess(self, x, candidate_beam, max_answer_length=100, **kwargs):
        """postprocess
        Postprocess to the inference_python engine output.
        Args:
            x: Inference engine output.
            candidate_beam (int): the number of candidate span index during postprocess.
            max_answer_length (int): the maximum length of answer.
        Returns:
            score: the score of the answer. (Larger = more confidence)
            answer_text (str): the answer text of the question.
        """
        index_offset = len(self.args.question) + 3

        result_start_logits, result_end_logits = x

        start_indexes = get_best_indexes(result_start_logits, candidate_beam)
        end_indexes = get_best_indexes(result_end_logits, candidate_beam)
        cls_token_score = result_start_logits[0] + result_end_logits[0]

        predictions = []

        for start_index in start_indexes:
            for end_index in end_indexes:
                if end_index < start_index:
                    continue
                length = end_index - start_index + 1
                if length > max_answer_length:
                    continue

                score = result_start_logits[start_index] + result_end_logits[end_index] - cls_token_score
                predictions.append(
                    (float(score), start_index, end_index))
        if not predictions:
            return None, None
        score, start_index, end_index = max(predictions, key=lambda x: x[0])
        answer_text = self.args.article[start_index - index_offset:end_index - index_offset + 1]
        return score, answer_text, start_index, end_index

    def run(self, input_ids, token_type_ids, attention_mask):
        """run
                Inference process using inference_python engine.
                Args:
                    input_ids (array): Input_ids after preprocess.
                    token_type_ids (array): ids that distinguish different input sequences.
                    attention_mask (array): 1/0 array that indicate the position of sample text.
                Returns: Inference engine output
        """
        self.input_tensors[0].copy_from_cpu(input_ids)
        self.input_tensors[1].copy_from_cpu(token_type_ids)
        self.input_tensors[2].copy_from_cpu(attention_mask)
        self.predictor.run()

        start_pos_logit = self.output_tensors[0].copy_to_cpu()
        end_pos_logit = self.output_tensors[1].copy_to_cpu()

        return start_pos_logit[0], end_pos_logit[0]


def get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""

    return np.argsort(logits[1:])[:-n_best_size - 1:-1] + 1


def get_args():
    import argparse
    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    parser = argparse.ArgumentParser()
    parser.add_argument("--article", default=test_article)
    parser.add_argument("--question", default=test_question)
    parser.add_argument("--model_dir", default="canine_infer")
    parser.add_argument("--use-gpu", default=False, type=str2bool, help="use_gpu")
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--warmup", default=0, type=int)
    parser.add_argument("--candidate_beam", default=5, type=int)
    parser.add_argument("--max_answer_length", type=int, default=100)
    parser.add_argument("--benchmark", default=False, type=str2bool, help="benchmark")
    return parser.parse_args()


def infer_main(args):
    """infer_main
    Main inference_python function.
    Args:
        args: Parameters generated using argparser.
    Returns:
        label_id: Class index of the input.
        prob: : Probability of the input.
    """
    inference_engine = InferenceEngine(args)

    # init benchmark
    if args.benchmark:
        import auto_log
        autolog = auto_log.AutoLogger(
            model_name="Canine MRC QA model",
            batch_size=args.batch_size,
            inference_config=inference_engine.config,
            gpu_ids="auto" if args.use_gpu else None)

    assert args.batch_size == 1, "batch size just supports 1 now."

    # enable benchmark
    if args.benchmark:
        autolog.times.start()

    # preprocess
    inputs = inference_engine.preprocess(args.article, args.question)

    if args.benchmark:
        autolog.times.stamp()

    output = inference_engine.run(**inputs)
    #
    if args.benchmark:
        autolog.times.stamp()
    #
    # postprocess
    score, answer, start_idx, end_idx = inference_engine.postprocess(output, candidate_beam=args.candidate_beam)
    #
    if args.benchmark:
        autolog.times.stamp()
        autolog.times.end(stamp=True)
        autolog.report()
    #
    print(f">>> Article: {args.article}\n"
          f">>> Question: {args.question}\n"
          f"\n>>> Answer Text: {answer}, score: {score}")
    # return label_id, prob
    return score, start_idx, end_idx


if __name__ == "__main__":
    args = get_args()
    score, start_idx, end_idx = infer_main(args)
    from reprod_log import ReprodLogger

    print(score, start_idx, end_idx)
    reprod_logger = ReprodLogger()
    reprod_logger.add("score", np.array([score]))
    reprod_logger.add("span", np.array([start_idx, end_idx]))
    reprod_logger.save("output_inference_engine.npy")
