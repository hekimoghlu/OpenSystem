/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 4, 2025.
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
#ifndef VPX_VPX_DSP_PPC_TXFM_COMMON_VSX_H_
#define VPX_VPX_DSP_PPC_TXFM_COMMON_VSX_H_

#include "vpx_dsp/ppc/types_vsx.h"

static const int32x4_t vec_dct_const_rounding = { 8192, 8192, 8192, 8192 };

static const uint32x4_t vec_dct_const_bits = { 14, 14, 14, 14 };

static const uint16x8_t vec_dct_scale_log2 = { 2, 2, 2, 2, 2, 2, 2, 2 };

static const int16x8_t cospi1_v = { 16364, 16364, 16364, 16364,
                                    16364, 16364, 16364, 16364 };
static const int16x8_t cospi2_v = { 16305, 16305, 16305, 16305,
                                    16305, 16305, 16305, 16305 };
static const int16x8_t cospi3_v = { 16207, 16207, 16207, 16207,
                                    16207, 16207, 16207, 16207 };
static const int16x8_t cospi4_v = { 16069, 16069, 16069, 16069,
                                    16069, 16069, 16069, 16069 };
static const int16x8_t cospi4m_v = { -16069, -16069, -16069, -16069,
                                     -16069, -16069, -16069, -16069 };
static const int16x8_t cospi5_v = { 15893, 15893, 15893, 15893,
                                    15893, 15893, 15893, 15893 };
static const int16x8_t cospi6_v = { 15679, 15679, 15679, 15679,
                                    15679, 15679, 15679, 15679 };
static const int16x8_t cospi7_v = { 15426, 15426, 15426, 15426,
                                    15426, 15426, 15426, 15426 };
static const int16x8_t cospi8_v = { 15137, 15137, 15137, 15137,
                                    15137, 15137, 15137, 15137 };
static const int16x8_t cospi8m_v = { -15137, -15137, -15137, -15137,
                                     -15137, -15137, -15137, -15137 };
static const int16x8_t cospi9_v = { 14811, 14811, 14811, 14811,
                                    14811, 14811, 14811, 14811 };
static const int16x8_t cospi10_v = { 14449, 14449, 14449, 14449,
                                     14449, 14449, 14449, 14449 };
static const int16x8_t cospi11_v = { 14053, 14053, 14053, 14053,
                                     14053, 14053, 14053, 14053 };
static const int16x8_t cospi12_v = { 13623, 13623, 13623, 13623,
                                     13623, 13623, 13623, 13623 };
static const int16x8_t cospi13_v = { 13160, 13160, 13160, 13160,
                                     13160, 13160, 13160, 13160 };
static const int16x8_t cospi14_v = { 12665, 12665, 12665, 12665,
                                     12665, 12665, 12665, 12665 };
static const int16x8_t cospi15_v = { 12140, 12140, 12140, 12140,
                                     12140, 12140, 12140, 12140 };
static const int16x8_t cospi16_v = { 11585, 11585, 11585, 11585,
                                     11585, 11585, 11585, 11585 };
static const int16x8_t cospi17_v = { 11003, 11003, 11003, 11003,
                                     11003, 11003, 11003, 11003 };
static const int16x8_t cospi18_v = { 10394, 10394, 10394, 10394,
                                     10394, 10394, 10394, 10394 };
static const int16x8_t cospi19_v = { 9760, 9760, 9760, 9760,
                                     9760, 9760, 9760, 9760 };
static const int16x8_t cospi20_v = { 9102, 9102, 9102, 9102,
                                     9102, 9102, 9102, 9102 };
static const int16x8_t cospi20m_v = { -9102, -9102, -9102, -9102,
                                      -9102, -9102, -9102, -9102 };
static const int16x8_t cospi21_v = { 8423, 8423, 8423, 8423,
                                     8423, 8423, 8423, 8423 };
static const int16x8_t cospi22_v = { 7723, 7723, 7723, 7723,
                                     7723, 7723, 7723, 7723 };
static const int16x8_t cospi23_v = { 7005, 7005, 7005, 7005,
                                     7005, 7005, 7005, 7005 };
static const int16x8_t cospi24_v = { 6270, 6270, 6270, 6270,
                                     6270, 6270, 6270, 6270 };
static const int16x8_t cospi25_v = { 5520, 5520, 5520, 5520,
                                     5520, 5520, 5520, 5520 };
static const int16x8_t cospi26_v = { 4756, 4756, 4756, 4756,
                                     4756, 4756, 4756, 4756 };
static const int16x8_t cospi27_v = { 3981, 3981, 3981, 3981,
                                     3981, 3981, 3981, 3981 };
static const int16x8_t cospi28_v = { 3196, 3196, 3196, 3196,
                                     3196, 3196, 3196, 3196 };
static const int16x8_t cospi29_v = { 2404, 2404, 2404, 2404,
                                     2404, 2404, 2404, 2404 };
static const int16x8_t cospi30_v = { 1606, 1606, 1606, 1606,
                                     1606, 1606, 1606, 1606 };
static const int16x8_t cospi31_v = { 804, 804, 804, 804, 804, 804, 804, 804 };

#endif  // VPX_VPX_DSP_PPC_TXFM_COMMON_VSX_H_
