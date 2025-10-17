/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 5, 2025.
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
#ifndef _CORECRYPTO_CCZP_H_
#define _CORECRYPTO_CCZP_H_

#include <corecrypto/ccn.h>

CC_PTRCHECK_CAPABLE_HEADER()

struct cczp;

typedef struct cczp *cczp_t;
typedef const struct cczp *cczp_const_t;

struct cczp_funcs;
typedef const struct cczp_funcs *cczp_funcs_t;

// keep cczp_hd and cczp structures consistent
// cczp_hd is typecasted to cczp to read EC curve params
// make sure n is the first element see ccrsa_ctx_n macro
#define __CCZP_HEADER_ELEMENTS_DEFINITIONS(pre) \
    cc_size pre##n;                             \
    cc_unit pre##bitlen;                        \
    cczp_funcs_t pre##funcs;

#define __CCZP_ELEMENTS_DEFINITIONS(pre)    \
    __CCZP_HEADER_ELEMENTS_DEFINITIONS(pre) \
    cc_unit pre##ccn[];

struct cczp {
    __CCZP_ELEMENTS_DEFINITIONS()
} CC_ALIGNED(CCN_UNIT_SIZE);

#define CCZP_N(ZP) ((ZP)->n)
#define CCZP_PRIME(ZP) ((ZP)->ccn)
#define CCZP_BITLEN(ZP) ((ZP)->bitlen)

CC_NONNULL((1))
cc_size cczp_n(cczp_const_t zp);

CC_NONNULL((1))
const cc_unit * cc_indexable cczp_prime(cczp_const_t zp);

CC_NONNULL((1))
size_t cczp_bitlen(cczp_const_t zp);

#endif /* _CORECRYPTO_CCZP_H_ */
