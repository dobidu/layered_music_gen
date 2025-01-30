# duration_validator.py
from dataclasses import dataclass
from typing import List, Dict, Optional, Set
from enum import Enum
import math
import logging

class NoteValue(Enum):
    """Valores rítmicos padrão."""
    WHOLE = 4.0          
    HALF = 2.0           
    QUARTER = 1.0        
    EIGHTH = 0.5         
    SIXTEENTH = 0.25     
    
    DOTTED_HALF = 3.0    
    DOTTED_QUARTER = 1.5 
    DOTTED_EIGHTH = 0.75 
    
    TRIPLET_QUARTER = 0.667
    TRIPLET_EIGHTH = 0.333

@dataclass
class TimeSignatureInfo:
    numerator: int
    denominator: int
    is_compound: bool
    valid_durations: Set[float]
    min_duration: float
    max_duration: float
    primary_division: float
    beats_per_measure: float

class DurationValidator:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._time_signature_cache: Dict[str, TimeSignatureInfo] = {}
    
    def _analyze_time_signature(self, time_signature: str) -> TimeSignatureInfo:
        """Analisa uma assinatura de tempo e retorna suas características."""
        if time_signature in self._time_signature_cache:
            return self._time_signature_cache[time_signature]
            
        numerator, denominator = map(int, time_signature.split('/'))
        is_compound = denominator == 8 and numerator % 3 == 0
        beats_per_measure = numerator / 3 if is_compound else numerator
        
        # Define durações válidas baseadas no tipo de compasso
        if is_compound:  # 6/8, 9/8, 12/8
            valid_durations = {
                NoteValue.DOTTED_QUARTER.value,
                NoteValue.QUARTER.value,
                NoteValue.DOTTED_EIGHTH.value,
                NoteValue.EIGHTH.value,
                NoteValue.SIXTEENTH.value
            }
            min_duration = NoteValue.SIXTEENTH.value
            max_duration = NoteValue.DOTTED_QUARTER.value * (numerator / 3)
            primary_division = 3.0
        else:  # 2/4, 3/4, 4/4
            valid_durations = {
                NoteValue.WHOLE.value,
                NoteValue.HALF.value,
                NoteValue.QUARTER.value,
                NoteValue.EIGHTH.value,
                NoteValue.SIXTEENTH.value,
                NoteValue.DOTTED_HALF.value,
                NoteValue.DOTTED_QUARTER.value,
                NoteValue.DOTTED_EIGHTH.value
            }
            min_duration = NoteValue.SIXTEENTH.value
            max_duration = float(numerator)
            primary_division = 2.0
        
        info = TimeSignatureInfo(
            numerator=numerator,
            denominator=denominator,
            is_compound=is_compound,
            valid_durations=valid_durations,
            min_duration=min_duration,
            max_duration=max_duration,
            primary_division=primary_division,
            beats_per_measure=beats_per_measure
        )
        
        self._time_signature_cache[time_signature] = info
        return info
    
    def get_valid_duration(self, duration: float, time_signature: str, 
                          remaining_beats: float, layer_type: str) -> float:
        """
        Retorna uma duração válida para o contexto específico.
        
        Args:
            duration: Duração proposta
            time_signature: Assinatura de tempo
            remaining_beats: Beats restantes no compasso
            layer_type: Tipo de camada ('melody', 'chord', 'bass', 'beat')
        """
        info = self._analyze_time_signature(time_signature)
        
        # Limita ao restante do compasso
        duration = min(duration, remaining_beats)
        
        # Ajusta baseado no tipo de camada
        if layer_type == 'melody':
            valid_durations = {
                NoteValue.QUARTER.value,
                NoteValue.EIGHTH.value,
                NoteValue.DOTTED_QUARTER.value
            } if info.is_compound else {
                NoteValue.HALF.value,
                NoteValue.QUARTER.value,
                NoteValue.EIGHTH.value
            }
        elif layer_type == 'chord':
            valid_durations = {
                NoteValue.DOTTED_QUARTER.value,
                NoteValue.DOTTED_HALF.value
            } if info.is_compound else {
                NoteValue.WHOLE.value,
                NoteValue.HALF.value,
                NoteValue.QUARTER.value
            }
        elif layer_type == 'bass':
            valid_durations = {
                NoteValue.DOTTED_QUARTER.value
            } if info.is_compound else {
                NoteValue.QUARTER.value,
                NoteValue.HALF.value
            }
        else:  # beat
            valid_durations = {
                NoteValue.EIGHTH.value
            } if info.is_compound else {
                NoteValue.QUARTER.value
            }
        
        # Encontra a duração válida mais próxima
        valid_duration = min(valid_durations, key=lambda x: abs(x - duration))
        
        return min(valid_duration, remaining_beats)
    
    def validate_layer_duration(self, durations: List[float], time_signature: str, 
                              layer_type: str) -> bool:
        """Valida as durações de uma camada completa."""
        info = self._analyze_time_signature(time_signature)
        total_duration = sum(durations)
        
        # Verifica se completa compassos inteiros
        if not math.isclose(total_duration % info.beats_per_measure, 0, abs_tol=0.001):
            self.logger.warning(f"{layer_type} layer duration does not complete full measures")
            
            return False
        
        return True
    
    def get_suggested_duration(self, time_signature: str, layer_type: str) -> float:
        """Retorna uma duração sugerida para o tipo de camada."""
        info = self._analyze_time_signature(time_signature)
        
        if layer_type == 'melody':
            return (NoteValue.DOTTED_QUARTER.value 
                   if info.is_compound else NoteValue.QUARTER.value)
        elif layer_type == 'chord':
            return (NoteValue.DOTTED_HALF.value 
                   if info.is_compound else NoteValue.HALF.value)
        elif layer_type == 'bass':
            return (NoteValue.DOTTED_QUARTER.value 
                   if info.is_compound else NoteValue.QUARTER.value)
        else:  # beat
            return (NoteValue.EIGHTH.value 
                   if info.is_compound else NoteValue.QUARTER.value)