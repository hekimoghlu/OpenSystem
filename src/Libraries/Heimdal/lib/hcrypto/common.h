/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 13, 2023.
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
#ifndef HCRYPTO_COMMON_H
#define HCRYPTO_COMMON_H 1

#ifdef NEED_CDSA

#include <Security/cssm.h>

extern const CSSM_DATA _hc_labelData;

CSSM_CSP_HANDLE
_hc_get_cdsa_csphandle(void);

#endif


int
_hc_BN_to_integer(BIGNUM *, heim_integer *);

BIGNUM *
_hc_integer_to_BN(const heim_integer *i, BIGNUM *bn);

#endif /* HCRYPTO_COMMON_H */
