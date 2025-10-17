/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 28, 2025.
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
// Some code came from common/rtcd.c in the WebM project.

#include "common_audio/signal_processing/include/signal_processing_library.h"

// TODO(bugs.webrtc.org/9553): These function pointers are useless. Refactor
// things so that we simply have a bunch of regular functions with different
// implementations for different platforms.

#if defined(WEBRTC_HAS_NEON)

const MaxAbsValueW16 WebRtcSpl_MaxAbsValueW16 = WebRtcSpl_MaxAbsValueW16Neon;
const MaxAbsValueW32 WebRtcSpl_MaxAbsValueW32 = WebRtcSpl_MaxAbsValueW32Neon;
const MaxValueW16 WebRtcSpl_MaxValueW16 = WebRtcSpl_MaxValueW16Neon;
const MaxValueW32 WebRtcSpl_MaxValueW32 = WebRtcSpl_MaxValueW32Neon;
const MinValueW16 WebRtcSpl_MinValueW16 = WebRtcSpl_MinValueW16Neon;
const MinValueW32 WebRtcSpl_MinValueW32 = WebRtcSpl_MinValueW32Neon;
const CrossCorrelation WebRtcSpl_CrossCorrelation =
    WebRtcSpl_CrossCorrelationNeon;
const DownsampleFast WebRtcSpl_DownsampleFast = WebRtcSpl_DownsampleFastNeon;
const ScaleAndAddVectorsWithRound WebRtcSpl_ScaleAndAddVectorsWithRound =
    WebRtcSpl_ScaleAndAddVectorsWithRoundC;

#elif defined(MIPS32_LE)

const MaxAbsValueW16 WebRtcSpl_MaxAbsValueW16 = WebRtcSpl_MaxAbsValueW16_mips;
const MaxAbsValueW32 WebRtcSpl_MaxAbsValueW32 =
#ifdef MIPS_DSP_R1_LE
    WebRtcSpl_MaxAbsValueW32_mips;
#else
    WebRtcSpl_MaxAbsValueW32C;
#endif
const MaxValueW16 WebRtcSpl_MaxValueW16 = WebRtcSpl_MaxValueW16_mips;
const MaxValueW32 WebRtcSpl_MaxValueW32 = WebRtcSpl_MaxValueW32_mips;
const MinValueW16 WebRtcSpl_MinValueW16 = WebRtcSpl_MinValueW16_mips;
const MinValueW32 WebRtcSpl_MinValueW32 = WebRtcSpl_MinValueW32_mips;
const CrossCorrelation WebRtcSpl_CrossCorrelation =
    WebRtcSpl_CrossCorrelation_mips;
const DownsampleFast WebRtcSpl_DownsampleFast = WebRtcSpl_DownsampleFast_mips;
const ScaleAndAddVectorsWithRound WebRtcSpl_ScaleAndAddVectorsWithRound =
#ifdef MIPS_DSP_R1_LE
    WebRtcSpl_ScaleAndAddVectorsWithRound_mips;
#else
    WebRtcSpl_ScaleAndAddVectorsWithRoundC;
#endif

#else

const MaxAbsValueW16 WebRtcSpl_MaxAbsValueW16 = WebRtcSpl_MaxAbsValueW16C;
const MaxAbsValueW32 WebRtcSpl_MaxAbsValueW32 = WebRtcSpl_MaxAbsValueW32C;
const MaxValueW16 WebRtcSpl_MaxValueW16 = WebRtcSpl_MaxValueW16C;
const MaxValueW32 WebRtcSpl_MaxValueW32 = WebRtcSpl_MaxValueW32C;
const MinValueW16 WebRtcSpl_MinValueW16 = WebRtcSpl_MinValueW16C;
const MinValueW32 WebRtcSpl_MinValueW32 = WebRtcSpl_MinValueW32C;
const CrossCorrelation WebRtcSpl_CrossCorrelation = WebRtcSpl_CrossCorrelationC;
const DownsampleFast WebRtcSpl_DownsampleFast = WebRtcSpl_DownsampleFastC;
const ScaleAndAddVectorsWithRound WebRtcSpl_ScaleAndAddVectorsWithRound =
    WebRtcSpl_ScaleAndAddVectorsWithRoundC;

#endif
