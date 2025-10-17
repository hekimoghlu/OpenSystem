/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 28, 2022.
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
#ifndef _CORECRYPTO_CC_FAULT_CANARY_H_
#define _CORECRYPTO_CC_FAULT_CANARY_H_

#include "cc.h"

CC_PTRCHECK_CAPABLE_HEADER()

#define CC_FAULT_CANARY_SIZE 16
typedef uint8_t cc_fault_canary_t[CC_FAULT_CANARY_SIZE];

extern const cc_fault_canary_t CCEC_FAULT_CANARY;
extern const cc_fault_canary_t CCRSA_PKCS1_FAULT_CANARY;
extern const cc_fault_canary_t CCRSA_PSS_FAULT_CANARY;

#define CC_FAULT_CANARY_MEMCPY(_dst_, _src_) cc_memcpy(_dst_, _src_, CC_FAULT_CANARY_SIZE)
#define CC_FAULT_CANARY_CLEAR(_name_) cc_memset(_name_, 0x00, CC_FAULT_CANARY_SIZE)
#define CC_FAULT_CANARY_EQUAL(_a_, _b_) (cc_cmp_safe(CC_FAULT_CANARY_SIZE, _a_, _b_) == 0)

#endif // _CORECRYPTO_CC_FAULT_CANARY_H_
