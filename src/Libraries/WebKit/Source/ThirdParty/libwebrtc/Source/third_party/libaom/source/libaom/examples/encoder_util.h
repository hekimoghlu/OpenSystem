/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 18, 2022.
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
// Utility functions used by encoder binaries.

#ifndef AOM_EXAMPLES_ENCODER_UTIL_H_
#define AOM_EXAMPLES_ENCODER_UTIL_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "aom/aom_image.h"

// Returns mismatch location (?loc[0],?loc[1]) and the values at that location
// in img1 (?loc[2]) and img2 (?loc[3]).
void aom_find_mismatch_high(const aom_image_t *const img1,
                            const aom_image_t *const img2, int yloc[4],
                            int uloc[4], int vloc[4]);

void aom_find_mismatch(const aom_image_t *const img1,
                       const aom_image_t *const img2, int yloc[4], int uloc[4],
                       int vloc[4]);

// Returns 1 if the two images match.
int aom_compare_img(const aom_image_t *const img1,
                    const aom_image_t *const img2);

#ifdef __cplusplus
}
#endif
#endif  // AOM_EXAMPLES_ENCODER_UTIL_H_
