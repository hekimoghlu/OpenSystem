/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 23, 2024.
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
/*
 *  ccErrors.h
 *  CommonCrypto
 */

#include <CommonCrypto/CommonCryptor.h>

#ifndef CCERRORS_H
#define CCERRORS_H

#define CONTEXT_SIZE_CHK(CCCTX,DIGESTDI) (sizeof(CCCTX) < ccdigest_di_size(DIGESTDI))
#define CC_NONULLPARM(X) if(NULL==(X)) return kCCParamError
#define CC_NONULLPARMRETNULL(X) if(NULL==(X)) return NULL

#define CC_FAILURE_LOG(fmt, ...)  os_log(OS_LOG_TYPE_FAULT, __FUNCTION__ ## "  " ## fmt , __VA_ARGS__)
#define CC_ABORT()      abort()


#endif // CCERRORS_H

