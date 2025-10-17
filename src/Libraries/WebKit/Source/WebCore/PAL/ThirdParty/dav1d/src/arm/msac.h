/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 23, 2022.
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
#ifndef DAV1D_SRC_ARM_MSAC_H
#define DAV1D_SRC_ARM_MSAC_H

unsigned dav1d_msac_decode_symbol_adapt4_neon(MsacContext *s, uint16_t *cdf,
                                              size_t n_symbols);
unsigned dav1d_msac_decode_symbol_adapt8_neon(MsacContext *s, uint16_t *cdf,
                                              size_t n_symbols);
unsigned dav1d_msac_decode_symbol_adapt16_neon(MsacContext *s, uint16_t *cdf,
                                               size_t n_symbols);
unsigned dav1d_msac_decode_hi_tok_neon(MsacContext *s, uint16_t *cdf);
unsigned dav1d_msac_decode_bool_adapt_neon(MsacContext *s, uint16_t *cdf);
unsigned dav1d_msac_decode_bool_equi_neon(MsacContext *s);
unsigned dav1d_msac_decode_bool_neon(MsacContext *s, unsigned f);

#if ARCH_AARCH64 || defined(__ARM_NEON)
#define dav1d_msac_decode_symbol_adapt4  dav1d_msac_decode_symbol_adapt4_neon
#define dav1d_msac_decode_symbol_adapt8  dav1d_msac_decode_symbol_adapt8_neon
#define dav1d_msac_decode_symbol_adapt16 dav1d_msac_decode_symbol_adapt16_neon
#define dav1d_msac_decode_hi_tok         dav1d_msac_decode_hi_tok_neon
#define dav1d_msac_decode_bool_adapt     dav1d_msac_decode_bool_adapt_neon
#define dav1d_msac_decode_bool_equi      dav1d_msac_decode_bool_equi_neon
#define dav1d_msac_decode_bool           dav1d_msac_decode_bool_neon
#endif

#endif /* DAV1D_SRC_ARM_MSAC_H */
