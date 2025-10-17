/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 18, 2022.
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
#include <emmintrin.h>  // SSE2

#include "config/aom_config.h"
#include "config/aom_dsp_rtcd.h"

#include "aom_dsp/aom_dsp_common.h"
#include "aom_dsp/x86/fwd_txfm_sse2.h"

#define DCT_HIGH_BIT_DEPTH 0
#define FDCT4x4_2D_HELPER fdct4x4_helper
#define FDCT4x4_2D aom_fdct4x4_sse2
#define FDCT4x4_2D_LP aom_fdct4x4_lp_sse2
#define FDCT8x8_2D aom_fdct8x8_sse2
#include "aom_dsp/x86/fwd_txfm_impl_sse2.h"
#undef FDCT4x4_2D_HELPER
#undef FDCT4x4_2D
#undef FDCT4x4_2D_LP
#undef FDCT8x8_2D

#if CONFIG_AV1_HIGHBITDEPTH

#undef DCT_HIGH_BIT_DEPTH
#define DCT_HIGH_BIT_DEPTH 1
#define FDCT8x8_2D aom_highbd_fdct8x8_sse2
#include "aom_dsp/x86/fwd_txfm_impl_sse2.h"  // NOLINT
#undef FDCT8x8_2D

#endif
