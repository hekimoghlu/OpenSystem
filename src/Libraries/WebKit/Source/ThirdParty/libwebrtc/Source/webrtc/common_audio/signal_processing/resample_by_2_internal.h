/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 6, 2024.
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
/*
 * This header file contains some internal resampling functions.
 *
 */

#ifndef COMMON_AUDIO_SIGNAL_PROCESSING_RESAMPLE_BY_2_INTERNAL_H_
#define COMMON_AUDIO_SIGNAL_PROCESSING_RESAMPLE_BY_2_INTERNAL_H_

#include <stdint.h>

/*******************************************************************
 * resample_by_2_fast.c
 * Functions for internal use in the other resample functions
 ******************************************************************/
void WebRtcSpl_DownBy2IntToShort(int32_t* in,
                                 int32_t len,
                                 int16_t* out,
                                 int32_t* state);

void WebRtcSpl_DownBy2ShortToInt(const int16_t* in,
                                 int32_t len,
                                 int32_t* out,
                                 int32_t* state);

void WebRtcSpl_UpBy2ShortToInt(const int16_t* in,
                               int32_t len,
                               int32_t* out,
                               int32_t* state);

void WebRtcSpl_UpBy2IntToInt(const int32_t* in,
                             int32_t len,
                             int32_t* out,
                             int32_t* state);

void WebRtcSpl_UpBy2IntToShort(const int32_t* in,
                               int32_t len,
                               int16_t* out,
                               int32_t* state);

void WebRtcSpl_LPBy2ShortToInt(const int16_t* in,
                               int32_t len,
                               int32_t* out,
                               int32_t* state);

void WebRtcSpl_LPBy2IntToInt(const int32_t* in,
                             int32_t len,
                             int32_t* out,
                             int32_t* state);

#endif  // COMMON_AUDIO_SIGNAL_PROCESSING_RESAMPLE_BY_2_INTERNAL_H_
