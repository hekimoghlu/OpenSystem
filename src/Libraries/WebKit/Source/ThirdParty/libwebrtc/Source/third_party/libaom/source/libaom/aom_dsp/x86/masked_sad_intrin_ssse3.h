/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 24, 2022.
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
#ifndef AOM_AOM_DSP_X86_MASKED_SAD_INTRIN_SSSE3_H_
#define AOM_AOM_DSP_X86_MASKED_SAD_INTRIN_SSSE3_H_

unsigned int aom_masked_sad8xh_ssse3(const uint8_t *src_ptr, int src_stride,
                                     const uint8_t *a_ptr, int a_stride,
                                     const uint8_t *b_ptr, int b_stride,
                                     const uint8_t *m_ptr, int m_stride,
                                     int height);

unsigned int aom_masked_sad4xh_ssse3(const uint8_t *src_ptr, int src_stride,
                                     const uint8_t *a_ptr, int a_stride,
                                     const uint8_t *b_ptr, int b_stride,
                                     const uint8_t *m_ptr, int m_stride,
                                     int height);

unsigned int aom_highbd_masked_sad4xh_ssse3(const uint8_t *src8, int src_stride,
                                            const uint8_t *a8, int a_stride,
                                            const uint8_t *b8, int b_stride,
                                            const uint8_t *m_ptr, int m_stride,
                                            int height);

#endif  // AOM_AOM_DSP_X86_MASKED_SAD_INTRIN_SSSE3_H_
