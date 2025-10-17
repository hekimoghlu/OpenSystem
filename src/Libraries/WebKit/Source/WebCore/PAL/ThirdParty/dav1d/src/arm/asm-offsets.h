/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 3, 2021.
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
#ifndef ARM_ASM_OFFSETS_H
#define ARM_ASM_OFFSETS_H

#define FGD_SEED                         0
#define FGD_AR_COEFF_LAG                 92
#define FGD_AR_COEFFS_Y                  96
#define FGD_AR_COEFFS_UV                 120
#define FGD_AR_COEFF_SHIFT               176
#define FGD_GRAIN_SCALE_SHIFT            184

#define FGD_SCALING_SHIFT                88
#define FGD_UV_MULT                      188
#define FGD_UV_LUMA_MULT                 196
#define FGD_UV_OFFSET                    204
#define FGD_CLIP_TO_RESTRICTED_RANGE     216

#endif /* ARM_ASM_OFFSETS_H */
