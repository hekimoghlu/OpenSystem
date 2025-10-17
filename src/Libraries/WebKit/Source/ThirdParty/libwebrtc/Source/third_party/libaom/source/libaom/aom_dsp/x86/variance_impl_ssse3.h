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
#ifndef AOM_AOM_DSP_X86_VARIANCE_IMPL_SSSE3_H_
#define AOM_AOM_DSP_X86_VARIANCE_IMPL_SSSE3_H_

#include <stdint.h>

void aom_var_filter_block2d_bil_first_pass_ssse3(
    const uint8_t *a, uint16_t *b, unsigned int src_pixels_per_line,
    unsigned int pixel_step, unsigned int output_height,
    unsigned int output_width, const uint8_t *filter);

void aom_var_filter_block2d_bil_second_pass_ssse3(
    const uint16_t *a, uint8_t *b, unsigned int src_pixels_per_line,
    unsigned int pixel_step, unsigned int output_height,
    unsigned int output_width, const uint8_t *filter);

#endif  // AOM_AOM_DSP_X86_VARIANCE_IMPL_SSSE3_H_
