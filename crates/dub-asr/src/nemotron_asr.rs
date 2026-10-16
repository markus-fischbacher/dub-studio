//! Nemotron 3.5 ASR (NVIDIA, 35 languages with punctuation) through parakeet-rs: the folder of its int8 export
//! (`encoder.onnx` with `encoder.onnx.data`, `decoder_joint.onnx`, `tokenizer.model`). The language is told to the
//! model; a recording is heard in windows cut at pauses, as Parakeet's.

use std::path::{Path, PathBuf};

use parakeet_rs::{Nemotron, TimestampMode};

use crate::segment::{segment_words, Segment, Word, SEG_MAX_DUR, SEG_MAX_GAP};
use crate::{append_dedup_words, ensure_ort_dylib, exec_config, load_wav_16k_mono, window, words_from_tokens, AsrEngine, AsrError, SpeakerSegment, Turn};

/// The model's name for a language: a bare code where it has one, its locale otherwise; `""` and `auto` leave the
/// choice to the model.
pub fn nemotron_lang(lang: &str) -> Option<&'static str> {
    Some(match lang.trim().to_ascii_lowercase().as_str() {
        "" | "auto" => "auto",
        "ar" => "ar",
        "bg" => "bg",
        "cs" => "cs",
        "da" => "da",
        "de" => "de",
        "el" => "el",
        "en" => "en",
        "es" => "es",
        "et" => "et",
        "fi" => "fi",
        "fr" => "fr",
        "he" => "he-IL",
        "hi" => "hi",
        "hr" => "hr",
        "hu" => "hu",
        "it" => "it",
        "ja" => "ja-JP",
        "ko" => "ko",
        "lt" => "lt",
        "lv" => "lv",
        "mt" => "mt-MT",
        "nl" => "nl",
        "no" | "nb" => "nb",
        "pl" => "pl",
        "pt" => "pt",
        "ro" => "ro",
        "ru" => "ru",
        "sk" => "sk",
        "sl" => "sl",
        "sv" => "sv",
        "th" => "th-TH",
        "tr" => "tr",
        "uk" => "uk",
        "vi" => "vi-VN",
        "zh" => "zh-CN",
        _ => return None,
    })
}

pub struct NemotronAsr {
    dir: PathBuf,
    model: Option<Nemotron>,
}

impl NemotronAsr {
    pub fn new(dir: impl AsRef<Path>) -> Self {
        Self { dir: dir.as_ref().to_path_buf(), model: None }
    }

    fn model(&mut self, lang: &str) -> Result<&mut Nemotron, AsrError> {
        let code = nemotron_lang(lang).ok_or_else(|| AsrError::Parakeet(format!("Nemotron 3.5 ASR does not know the language {lang}")))?;
        if self.model.is_none() {
            ensure_ort_dylib();
            self.model = Some(Nemotron::from_pretrained(&self.dir, Some(exec_config())).map_err(|e| AsrError::Parakeet(e.to_string()))?);
        }
        let model = self.model.as_mut().expect("loaded above");
        model.set_target_lang(code).map_err(|e| AsrError::Parakeet(e.to_string()))?;
        Ok(model)
    }

    fn words(&mut self, audio: &[f32], sr: u32, lang: &str) -> Result<Vec<Word>, AsrError> {
        let res = self
            .model(lang)?
            .transcribe_audio_with_timestamps(audio, Some(TimestampMode::Words))
            .map_err(|e| AsrError::Parakeet(e.to_string()))?;
        Ok(words_from_tokens(res.tokens, 0.0, audio.len() as f64 / sr as f64))
    }

    fn words_windowed(&mut self, audio: &[f32], sr: u32, lang: &str) -> Result<Vec<Word>, AsrError> {
        let cfg = window::WindowConfig::default();
        let windows = window::plan_windows(audio, sr, audio.len() as f64 / sr as f64, &cfg);
        if windows.len() <= 1 {
            return self.words(audio, sr, lang);
        }
        let mut all: Vec<Word> = Vec::new();
        for w in &windows {
            let a = ((w.start * sr as f64) as usize).min(audio.len());
            let b = ((w.end * sr as f64) as usize).min(audio.len());
            if b <= a {
                continue;
            }
            let mut words = self.words(&audio[a..b], sr, lang)?;
            for word in &mut words {
                word.start += w.offset();
                word.end += w.offset();
            }
            append_dedup_words(&mut all, words, w.start);
        }
        Ok(all)
    }
}

impl AsrEngine for NemotronAsr {
    fn transcribe(&mut self, wav: &Path, lang: &str) -> Result<Vec<Segment>, AsrError> {
        let (audio, sr) = load_wav_16k_mono(wav)?;
        let words = self.words_windowed(&audio, sr, lang)?;
        Ok(segment_words(&words, SEG_MAX_GAP, SEG_MAX_DUR))
    }

    fn transcribe_turns(&mut self, wav: &Path, turns: &[Turn], lang: &str) -> Result<Vec<SpeakerSegment>, AsrError> {
        let (audio, sr) = load_wav_16k_mono(wav)?;
        let min_len = (0.2 * sr as f64) as usize;
        let mut out = Vec::new();
        for t in turns {
            let a = (t.start * sr as f64) as usize;
            let b = ((t.end * sr as f64) as usize).min(audio.len());
            if b <= a || (b - a) < min_len {
                continue;
            }
            let words = self.words(&audio[a..b], sr, lang)?;
            for s in segment_words(&words, SEG_MAX_GAP, SEG_MAX_DUR) {
                out.push(SpeakerSegment { start: t.start + s.start, end: t.start + s.end, text: s.text, speaker: t.speaker });
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn languages_are_named_as_the_model_knows_them() {
        assert_eq!(nemotron_lang("ru"), Some("ru"));
        assert_eq!(nemotron_lang("zh"), Some("zh-CN"));
        assert_eq!(nemotron_lang("JA"), Some("ja-JP"));
        assert_eq!(nemotron_lang("auto"), Some("auto"));
        assert_eq!(nemotron_lang("sw"), None);
    }
}
