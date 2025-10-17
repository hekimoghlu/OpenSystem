/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 11, 2023.
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
#include <stddef.h>
#include <stdint.h>

#ifndef DAV1D_SRC_ITX_1D_H
#define DAV1D_SRC_ITX_1D_H

#define decl_itx_1d_fn(name) \
void (name)(int32_t *c, ptrdiff_t stride, int min, int max)
typedef decl_itx_1d_fn(*itx_1d_fn);

decl_itx_1d_fn(dav1d_inv_dct4_1d_c);
decl_itx_1d_fn(dav1d_inv_dct8_1d_c);
decl_itx_1d_fn(dav1d_inv_dct16_1d_c);
decl_itx_1d_fn(dav1d_inv_dct32_1d_c);
decl_itx_1d_fn(dav1d_inv_dct64_1d_c);

decl_itx_1d_fn(dav1d_inv_adst4_1d_c);
decl_itx_1d_fn(dav1d_inv_adst8_1d_c);
decl_itx_1d_fn(dav1d_inv_adst16_1d_c);

decl_itx_1d_fn(dav1d_inv_flipadst4_1d_c);
decl_itx_1d_fn(dav1d_inv_flipadst8_1d_c);
decl_itx_1d_fn(dav1d_inv_flipadst16_1d_c);

decl_itx_1d_fn(dav1d_inv_identity4_1d_c);
decl_itx_1d_fn(dav1d_inv_identity8_1d_c);
decl_itx_1d_fn(dav1d_inv_identity16_1d_c);
decl_itx_1d_fn(dav1d_inv_identity32_1d_c);

void dav1d_inv_wht4_1d_c(int32_t *c, ptrdiff_t stride);

#endif /* DAV1D_SRC_ITX_1D_H */
