/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 7, 2021.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#ifdef FIXED_POINT
#include "main_FIX.h"
#else
#include "main_FLP.h"
#endif
#include "tuning_parameters.h"

/* High-pass filter with cutoff frequency adaptation based on pitch lag statistics */
void silk_HP_variable_cutoff(
    silk_encoder_state_Fxx          state_Fxx[]                         /* I/O  Encoder states                              */
)
{
   opus_int   quality_Q15;
   opus_int32 pitch_freq_Hz_Q16, pitch_freq_log_Q7, delta_freq_Q7;
   silk_encoder_state *psEncC1 = &state_Fxx[ 0 ].sCmn;

   /* Adaptive cutoff frequency: estimate low end of pitch frequency range */
   if( psEncC1->prevSignalType == TYPE_VOICED ) {
      /* difference, in log domain */
      pitch_freq_Hz_Q16 = silk_DIV32_16( silk_LSHIFT( silk_MUL( psEncC1->fs_kHz, 1000 ), 16 ), psEncC1->prevLag );
      pitch_freq_log_Q7 = silk_lin2log( pitch_freq_Hz_Q16 ) - ( 16 << 7 );

      /* adjustment based on quality */
      quality_Q15 = psEncC1->input_quality_bands_Q15[ 0 ];
      pitch_freq_log_Q7 = silk_SMLAWB( pitch_freq_log_Q7, silk_SMULWB( silk_LSHIFT( -quality_Q15, 2 ), quality_Q15 ),
            pitch_freq_log_Q7 - ( silk_lin2log( SILK_FIX_CONST( VARIABLE_HP_MIN_CUTOFF_HZ, 16 ) ) - ( 16 << 7 ) ) );

      /* delta_freq = pitch_freq_log - psEnc->variable_HP_smth1; */
      delta_freq_Q7 = pitch_freq_log_Q7 - silk_RSHIFT( psEncC1->variable_HP_smth1_Q15, 8 );
      if( delta_freq_Q7 < 0 ) {
         /* less smoothing for decreasing pitch frequency, to track something close to the minimum */
         delta_freq_Q7 = silk_MUL( delta_freq_Q7, 3 );
      }

      /* limit delta, to reduce impact of outliers in pitch estimation */
      delta_freq_Q7 = silk_LIMIT_32( delta_freq_Q7, -SILK_FIX_CONST( VARIABLE_HP_MAX_DELTA_FREQ, 7 ), SILK_FIX_CONST( VARIABLE_HP_MAX_DELTA_FREQ, 7 ) );

      /* update smoother */
      psEncC1->variable_HP_smth1_Q15 = silk_SMLAWB( psEncC1->variable_HP_smth1_Q15,
            silk_SMULBB( psEncC1->speech_activity_Q8, delta_freq_Q7 ), SILK_FIX_CONST( VARIABLE_HP_SMTH_COEF1, 16 ) );

      /* limit frequency range */
      psEncC1->variable_HP_smth1_Q15 = silk_LIMIT_32( psEncC1->variable_HP_smth1_Q15,
            silk_LSHIFT( silk_lin2log( VARIABLE_HP_MIN_CUTOFF_HZ ), 8 ),
            silk_LSHIFT( silk_lin2log( VARIABLE_HP_MAX_CUTOFF_HZ ), 8 ) );
   }
}
