/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 17, 2024.
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
#include "vpx_config.h"
#include "vp8_rtcd.h"
#include "vpx_ports/x86.h"
#include "vp8/encoder/block.h"

int vp8_mbblock_error_sse2_impl(short *coeff_ptr, short *dcoef_ptr, int dc);
int vp8_mbblock_error_sse2(MACROBLOCK *mb, int dc) {
  short *coeff_ptr = mb->block[0].coeff;
  short *dcoef_ptr = mb->e_mbd.block[0].dqcoeff;
  return vp8_mbblock_error_sse2_impl(coeff_ptr, dcoef_ptr, dc);
}

int vp8_mbuverror_sse2_impl(short *s_ptr, short *d_ptr);
int vp8_mbuverror_sse2(MACROBLOCK *mb) {
  short *s_ptr = &mb->coeff[256];
  short *d_ptr = &mb->e_mbd.dqcoeff[256];
  return vp8_mbuverror_sse2_impl(s_ptr, d_ptr);
}
