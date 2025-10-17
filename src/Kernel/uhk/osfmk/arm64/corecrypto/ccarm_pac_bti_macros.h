/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 5, 2023.
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
#ifndef _CORECRYPTO_CCARM_PAC_BTI_MACROS_H_
#define _CORECRYPTO_CCARM_PAC_BTI_MACROS_H_

/*
 * This file defines commonly used macros in handwritten assembly
 * for making functions BTI and PAC compatible.
 */

#ifndef __arm64e__
#define __arm64e__ 0
#endif

.macro SIGN_LR
#if __arm64e__
pacibsp
#endif
.endmacro

.macro AUTH_LR_AND_RET
#if __arm64e__
retab
#else
ret
#endif
.endmacro

.macro BRANCH_TARGET_CALL
#if __arm64e__
hint #34         /* bti c */
#endif
.endmacro



#endif /* _CORECRYPTO_CCARM_PAC_BTI_MACROS_H_ */
