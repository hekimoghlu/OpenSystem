/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 7, 2022.
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
 *  ECBTest.c
 *  CommonCrypto
 */

#include <stdio.h>
#include "CCCryptorTestFuncs.h"
#include "testbyteBuffer.h"
#include "testmore.h"
#include "capabilities.h"

#if (CCSYMECB == 0)
entryPoint(CommonCryptoSymECB,"CommonCrypto Symmetric ECB Testing")
#else


static int kTestTestCount = 4;

int CommonCryptoSymECB(int __unused argc, char *const * __unused argv) {
	char *keyStr;
	char *iv;
	char *plainText;
	char *cipherText;
    CCAlgorithm alg;
    CCOptions options;
	int retval;
    int accum = 0;

	keyStr 	   = "000102030405060708090a0b0c0d0e0f";
    alg		   = kCCAlgorithmAES128;
    options    = kCCOptionECBMode;
    
	plan_tests(kTestTestCount);
    
    accum = (int) genRandomSize(1,10);
    
    iv = NULL;

	// 16
    plainText  = "0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a";
	cipherText = "d307b25d3abaf87c0053e8188152992a";
    retval = CCCryptTestCase(keyStr, iv, alg, options, cipherText, plainText, true);
    ok(retval == 0, "ECB with Padding 16 byte CCCrypt NULL IV");
    retval = CCMultiCryptTestCase(keyStr, iv, alg, options, cipherText, plainText);
    ok(retval == 0, "ECB with Padding 16 byte Multiple Updates NULL IV");
    accum |= retval;

	// 32
	plainText  = "0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a";
	cipherText = "d307b25d3abaf87c0053e8188152992ad307b25d3abaf87c0053e8188152992a";
    retval = CCCryptTestCase(keyStr, iv, alg, options, cipherText, plainText, true);
    ok(retval == 0, "ECB 32 byte CCCrypt NULL IV");
    accum |= retval;
    retval = CCMultiCryptTestCase(keyStr, iv, alg, options, cipherText, plainText);
    ok(retval == 0, "ECB 32 byte Multiple Updates NULL IV");
    accum |= retval;

    return accum != 0;
}
#endif

