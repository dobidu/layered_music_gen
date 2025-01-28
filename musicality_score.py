import sys
import librosa
import numpy as np
from scipy.stats import entropy
from enum import Enum
import logging
import warnings
from typing import Tuple, Dict, Optional

class MusicalityAnalyzer:
    def __init__(self):
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Component weights
        self.weights = {
            'tempo': 0.25,
            'harmony': 0.25,
            'rhythm': 0.25,
            'timbre': 0.15,
            'noise': 0.10
        }

        # Classification thresholds
        self.thresholds = {
            'high': 0.65,
            'medium': 0.45,
            'low': 0.25
        }

        # Calibration factors
        self.calibration = {
            'tempo': 1.1,
            'harmony': 1.1,
            'rhythm': 1.0,
            'noise': 0.9
        }

    def analyze_tempo(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Analyze tempo characteristics."""
        try:
            # Get tempo and beats
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            
            # Calculate tempo stability
            if len(beats) >= 2:
                beat_intervals = np.diff(beats)
                tempo_stability = 1.0 / (1.0 + np.std(beat_intervals) / np.mean(beat_intervals))
            else:
                tempo_stability = 0

            # Calculate tempo in reasonable range (60-180 BPM)
            tempo_range = np.clip((tempo - 60) / (180 - 60), 0, 1)
            tempo_reasonableness = 1.0 - abs(tempo_range - 0.5) * 2

            # Get onset strength
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            tempo_clarity = np.mean(onset_env) / np.max(onset_env) if len(onset_env) > 0 else 0

            return {
                'stability': tempo_stability,
                'reasonableness': tempo_reasonableness,
                'clarity': tempo_clarity
            }
        except Exception as e:
            self.logger.error(f"Error in tempo analysis: {str(e)}")
            return {'stability': 0, 'reasonableness': 0, 'clarity': 0}

    def analyze_harmony(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Analyze harmonic content."""
        try:
            # Chromagram for harmony analysis
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr, bins_per_octave=36)
            
            # Tonnetz features
            tonnetz = librosa.feature.tonnetz(y=y, sr=sr)

            # Key clarity
            key_clarity = np.max(np.mean(chroma, axis=1))

            # Harmonic changes
            harmonic_changes = np.mean(np.diff(chroma, axis=1) != 0)

            # Consonance measure
            consonance = np.mean([np.corrcoef(tonnetz[i], tonnetz[i+1])[0,1] 
                                for i in range(len(tonnetz)-1)])

            return {
                'key_clarity': key_clarity,
                'stability': 1 - harmonic_changes,
                'consonance': max(0, consonance)
            }
        except Exception as e:
            self.logger.error(f"Error in harmony analysis: {str(e)}")
            return {'key_clarity': 0, 'stability': 0, 'consonance': 0}

    def analyze_rhythm(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Analyze rhythmic features with balanced metrics."""
        try:
            # 1. Onset detection com configuração mais equilibrada
            onset_env = librosa.onset.onset_strength(
                y=y, 
                sr=sr,
                aggregate=np.mean,
                n_mels=128
            )
            
            # 2. Beat tracking com parâmetros mais flexíveis
            tempo, beats = librosa.beat.beat_track(
                onset_envelope=onset_env, 
                sr=sr,
                hop_length=512,
                tightness=50
            )
            
            # 3. Rhythm regularity com tolerância maior
            if len(beats) >= 4:
                beat_intervals = np.diff(beats)
                local_variance = np.std(beat_intervals) / (np.mean(beat_intervals) + 1e-6)
                global_variance = np.std(beat_intervals) / (len(beat_intervals) + 1e-6)
                rhythm_regularity = 1.0 / (1.0 + 0.5 * (local_variance + global_variance))
            else:
                rhythm_regularity = 0.0

            # 4. Beat strength com normalização mais suave
            if len(beats) > 0:
                peak_values = onset_env[beats]
                background = np.mean(onset_env)
                beat_strength = np.clip((np.mean(peak_values) - background) / 
                                      (np.max(onset_env) + 1e-6), 0, 1)
            else:
                beat_strength = 0.0

            # 5. Pattern analysis mais tolerante
            if len(onset_env) > 0:
                acf = librosa.autocorrelate(onset_env)
                acf = acf[:len(acf)//2]
                pattern_score = np.clip(1.0 - 0.5 * (np.std(acf) / 
                                                    (np.mean(acf) + 1e-6)), 0, 1)
            else:
                pattern_score = 0.0

            # 6. Rhythm density com range mais amplo
            if len(y) > 0:
                density = len(beats) / (len(y) / sr)
                optimal_density = 1.5
                density_score = np.clip(1.0 - 0.5 * abs(density - optimal_density) / 
                                      optimal_density, 0, 1)
            else:
                density_score = 0.0

            # 7. Rhythm complexity com tolerância maior
            if len(onset_env) > 1:
                onset_diffs = np.diff(onset_env)
                complexity = np.std(onset_diffs) / (np.mean(onset_env) + 1e-6)
                complexity_score = np.clip(1.0 - 0.5 * abs(complexity - 0.5), 0, 1)
            else:
                complexity_score = 0.0

            # Pesos ajustados e normalização mais equilibrada
            rhythm_scores = {
                'regularity': rhythm_regularity * 0.3,
                'strength': beat_strength * 0.25,
                'pattern': pattern_score * 0.2,
                'density': density_score * 0.15,
                'complexity': complexity_score * 0.1
            }

            # Normalização final mais suave
            return {k: np.clip(v, 0, 1) for k, v in rhythm_scores.items()}

        except Exception as e:
            self.logger.error(f"Error in rhythm analysis: {str(e)}")
            return {
                'regularity': 0,
                'strength': 0,
                'pattern': 0,
                'density': 0,
                'complexity': 0
            }

    def analyze_noise(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Analyze noise characteristics."""
        try:
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(y)
            zcr_score = 1.0 - np.mean(zcr)

            # Spectral flatness
            flatness = librosa.feature.spectral_flatness(y=y)
            flatness_score = 1.0 - np.mean(flatness)

            # Spectral contrast
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            contrast_score = np.mean(contrast) / np.max(contrast) if np.max(contrast) > 0 else 0

            # Combined noise score
            noise_score = (zcr_score + flatness_score + contrast_score) / 3

            return {
                'noise_level': 1.0 - noise_score,
                'music_signal_ratio': noise_score
            }
        except Exception as e:
            self.logger.error(f"Error in noise analysis: {str(e)}")
            return {'noise_level': 1.0, 'music_signal_ratio': 0.0}

    def calculate_musicality(self, filename: str) -> Tuple[float, Dict[str, float]]:
        """Calculate overall musicality score and classification."""
        try:
            # Load audio file
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                y, sr = librosa.load(filename)

            # Get component analyses
            tempo_scores = self.analyze_tempo(y, sr)
            harmony_scores = self.analyze_harmony(y, sr)
            rhythm_scores = self.analyze_rhythm(y, sr)
            noise_scores = self.analyze_noise(y, sr)

            # Ensure all values are scalar
            tempo_scores = {k: (np.mean(v) if isinstance(v, np.ndarray) else v) for k, v in tempo_scores.items()}
            harmony_scores = {k: (np.mean(v) if isinstance(v, np.ndarray) else v) for k, v in harmony_scores.items()}
            rhythm_scores = {k: (np.mean(v) if isinstance(v, np.ndarray) else v) for k, v in rhythm_scores.items()}
            noise_scores = {k: (np.mean(v) if isinstance(v, np.ndarray) else v) for k, v in noise_scores.items()}

            # Debugging: Print shapes of the outputs
            # print(f"Tempo scores: {tempo_scores}")
            # print(f"Harmony scores: {harmony_scores}")
            # print(f"Rhythm scores: {rhythm_scores}")
            # print(f"Noise scores: {noise_scores}")

            # Calculate component scores with calibration
            scores = {
                'tempo': np.mean(list(tempo_scores.values())) * self.calibration['tempo'],
                'harmony': np.mean(list(harmony_scores.values())) * self.calibration['harmony'],
                'rhythm': np.mean(list(rhythm_scores.values())) * self.calibration['rhythm'],
                'noise': noise_scores['music_signal_ratio'] * self.calibration['noise']
            }

            # Clip scores to [0, 1] range
            scores = {k: np.clip(v, 0, 1) for k, v in scores.items()}

            return np.mean(list(scores.values())), scores

        except Exception as e:
            self.logger.error(f"Error processing file {filename}: {str(e)}")
            return 0.0, {}

def get_musicality_score(filename: str) -> Tuple[float, Dict[str, float]]:
    """Main function to get musicality score."""
    analyzer = MusicalityAnalyzer()
    return analyzer.calculate_musicality(filename)

def main():
    if len(sys.argv) < 2:
        print('Usage: python musicality_score_ii.py <audio_file>')
        sys.exit(1)
    
    filename = sys.argv[1]
    score, component_scores = get_musicality_score(filename)
    
    print('\nMusicality Analysis Results:')
    print('-' * 40)
    print(f'Overall Score: {score:.2f}')
    print('\nComponent Scores:')
    for component, value in component_scores.items():
        print(f'{component.title():>10}: {value:.2f}')

if __name__ == '__main__':
    main()