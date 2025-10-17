/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 25, 2024.
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

//
//  testutil.h
//  CommonCrypto
//
//  Created by Richard Murphy on 1/21/14.
//  Copyright (c) 2014 Apple Inc. All rights reserved.
//

#include <stdio.h>
#include <string.h>
#include "testmore.h"
#include "testbyteBuffer.h"
#define COMMON_DIGEST_FOR_RFC_1321

#include <CommonCrypto/CommonDigestSPI.h>
#include <CommonCrypto/CommonHMAC.h>
#include <CommonCrypto/CommonKeyDerivation.h>

#ifndef CommonCrypto_testutil_h
#define CommonCrypto_testutil_h

int expectedEqualsComputed(char *label, byteBuffer expected, byteBuffer computed);
char *digestName(CCDigestAlgorithm digestSelector);

#define HMAC_UNIMP 99
CCHmacAlgorithm digestID2HMacID(CCDigestAlgorithm digestSelector);
CCPseudoRandomAlgorithm digestID2PRF(CCDigestAlgorithm digestSelector);



static inline byteBuffer mallocDigestByteBuffer(CCDigestAlgorithm alg) {
    return mallocByteBuffer(CCDigestGetOutputSize(alg));
}

#endif
