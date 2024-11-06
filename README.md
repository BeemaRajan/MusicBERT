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

Motivating question:
Can we leverage transformers (successful in NLP) for tasks in symbolic music understanding, such as sentiment analysis, classification, and sequence completion?
