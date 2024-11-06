# MusicBERT: Symbolic Music Understanding with Large-Scale Pre-Training

Mingliang Zeng, Xu Tan, Rui Wang, Zeqian Ju, Tao Qin, Tie-Yan Liu

Microsoft Research Asia

[Link to the MusicBERT paper](https://arxiv.org/abs/2106.05630)

# Introduction: Music is a language
Music, much like spoken or written language, has its own structure, syntax, and expressive power. Just as sentences are composed of words with specific grammatical rules, musical compositions are built from sequences of notes that adhere to harmonic, rhythmic, and structural patterns. These musical patterns, while often subconscious to listeners, convey emotions, ideas, and moods in ways that feel both universal and deeply personal.

In the field of Natural Language Processing (NLP), transformers have revolutionized how we process and understand language by analyzing and predicting word patterns. Given the structural similarities between music and language, this raises an intriguing question: 

***If transformers can analyze and generate language, can they also be used to analyze and generate music?***

Music, like text, can be broken down into smaller units (such as notes or chords), processed as sequences, and analyzed for patterns, emotions, and styles. This paper explores the exciting potential of using transformer models, widely successful in NLP, to gain deeper insights into music.
If transformers can understand language by predicting and classifying words, can they do the same for music?

Question 1: What does symbolic music look like?

hint: Think about how you might "read" music.

## Symbolic Music

![Sheet Music](https://upload.wikimedia.org/wikipedia/commons/thumb/8/8b/Oregon%2C_My_Oregon.jpg/330px-Oregon%2C_My_Oregon.jpg)

To understand how transformers can be applied to music, it’s important to distinguish between two types of music data: symbolic and audio. Audio recordings capture sound waves, preserving the nuances of performance, but they are complex and unstructured in a way that’s challenging for direct sequence modeling. Symbolic music, on the other hand, provides a structured representation of musical information, allowing models to process musical patterns more directly, similar to how text tokens are handled in NLP.

The most traditional form of symbolic music is sheet music—a visual score with notes, rhythms, and dynamics that musicians interpret to perform a piece. For digital music processing, however, pianoroll and MIDI formats are more commonly used. Pianoroll representation resembles a grid, with each row representing a pitch and each column representing a time slice, indicating which notes are played at specific moments. MIDI (Musical Instrument Digital Interface) goes even further, encoding not only pitch and rhythm but also attributes like duration, instrument, and velocity (loudness).

These symbolic formats allow us to tokenize music in ways that are compatible with transformer models. MIDI, especially, is highly structured, making it an ideal format for tokenization and sequence modeling—similar to the token-based representation in NLP. By converting music into a structured sequence of tokens, we can apply the same types of computational analysis to music as we do with language.

## Motivating Question

This paper addresses a core research question: Can we use transformers that excel at tasks like sentiment analysis, classification, and fill-in-the-blank predictions in text, to achieve similar results in symbolic music? Transformers in NLP handle tasks such as:

* Sentiment Analysis: Determining the emotional tone of text. In music, could we analyze mood, intensity, or emotional impact?
* Classification: Categorizing text by topic. Could we classify music into genres or styles using similar techniques?
* Fill-in-the-Blank (Masking): Predicting missing words in a sentence. In music, could we predict missing notes in a melody, or even suggest accompaniment?
Applying transformers to symbolic music isn’t a simple task due to music’s unique features, such as pitch, rhythm, and hierarchical structures that differ from natural language. MusicBERT tackles these challenges by adapting the transformer model specifically for symbolic music through OctupleMIDI encoding and a bar-level masking strategy. These innovations help the model understand and predict musical sequences more effectively, making it possible to apply transformer-based tasks to symbolic music in a way that could transform music analysis, generation, and understanding.

# Paper Overview
## How is music encoded?

In the MusicBERT paper, the authors discuss previous attempts to apply NLP techniques to MIDI and symbolic music, specifically highlighting two models:

* REMI (REpresentation of MIcro-timing): Introduced by Huang and Yang (2020), REMI encodes music in a way that captures temporal information by including bar, position, note duration, chord, and tempo. This model uses multiple tokens to represent various attributes of a single note, such as pitch and timing information, which allows it to capture expressive details. However, REMI's encoding sequence can become quite long, which poses challenges for efficient training and processing with transformers​​.
* CP (Compound Word): Developed by Hsiao et al. (2021), CP represents musical notes by compressing multiple attributes into a single token, reducing redundancy. The CP encoding combines information about pitch, duration, and velocity into one compound word, thereby shortening the sequence length compared to REMI. This compact approach is more efficient for transformer-based models, though it may lose some expressive details​​.

How is OctoMIDI different?
