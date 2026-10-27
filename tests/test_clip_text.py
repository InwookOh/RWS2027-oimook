from pathlib import Path
import unittest

import torch

from upar.clip_text import CLIPTextTransformer, Tokenizer, openai_text_state

VOCAB = Path('outputs/pretrained/clip_bpe/bpe_simple_vocab_16e6.txt.gz')


class ClipTextTests(unittest.TestCase):
    @unittest.skipUnless(VOCAB.exists(), 'CLIP BPE vocabulary not downloaded')
    def test_tokenizer_matches_openai_ids(self):
        tokens = Tokenizer(VOCAB)(['a photo of a cat'])
        # OpenAI clip.tokenize('a photo of a cat')
        self.assertEqual(tokens[0, :8].tolist(), [49406, 320, 1125, 539, 320, 2368, 49407, 0])

    def test_state_mapping_loads_strictly(self):
        torch.manual_seed(0)
        reference = CLIPTextTransformer(width=16, layers=2, heads=2, context=8, vocab=50)
        for parameter in reference.parameters():
            torch.nn.init.normal_(parameter, std=0.2)
        state = {'visual.proj': torch.zeros(1)}  # image-tower keys are ignored
        for key, value in reference.state_dict().items():
            if key.startswith('resblocks.'):
                key = 'transformer.' + key.replace('mlp.0', 'mlp.c_fc').replace('mlp.2', 'mlp.c_proj')
            state[key] = value
        model = CLIPTextTransformer(width=16, layers=2, heads=2, context=8, vocab=50)
        model.load_state_dict(openai_text_state(state), strict=True)
        tokens = torch.tensor([[1, 5, 49, 0, 0, 0, 0, 0], [1, 7, 3, 49, 0, 0, 0, 0]])
        output = model(tokens)
        self.assertTrue(torch.isfinite(output).all())
        torch.testing.assert_close(output, reference(tokens))
        # Causal: tokens after the end-of-text position do not change the sentence embedding.
        torch.testing.assert_close(model(torch.tensor([[1, 5, 49, 9, 9, 9, 9, 9]])), output[:1])


if __name__ == '__main__':
    unittest.main()
