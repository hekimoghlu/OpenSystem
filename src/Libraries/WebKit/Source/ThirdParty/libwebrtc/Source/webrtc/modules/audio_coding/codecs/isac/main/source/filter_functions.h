/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 18, 2025.
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
#ifndef MODULES_AUDIO_CODING_CODECS_ISAC_MAIN_SOURCE_FILTER_FUNCTIONS_H_
#define MODULES_AUDIO_CODING_CODECS_ISAC_MAIN_SOURCE_FILTER_FUNCTIONS_H_

#include <stddef.h>

#include "modules/audio_coding/codecs/isac/main/source/structs.h"

void WebRtcIsac_AutoCorr(double* r, const double* x, size_t N, size_t order);

void WebRtcIsac_WeightingFilter(const double* in,
                                double* weiout,
                                double* whiout,
                                WeightFiltstr* wfdata);

#endif  // MODULES_AUDIO_CODING_CODECS_ISAC_MAIN_SOURCE_FILTER_FUNCTIONS_H_
