/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 17, 2022.
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
#include <stdio.h>
#include "CCCryptorTestFuncs.h"
#include "testbyteBuffer.h"
#include "testmore.h"
#include "capabilities.h"

#if (CCBLOWFISH == 0)
entryPoint(CommonCryptoBlowfish,"CommonCrypto Blowfish Testing")
#else

static int kTestTestCount = 1;

int CommonCryptoBlowfish(int __unused argc, char *const * __unused argv)
{
    char *keyStr;
	char *iv = NULL;
	char *plainText;
	char *cipherText;
    CCMode mode;
    CCAlgorithm alg;
    CCPadding padding;
    int retval, accum = 0;
	keyStr 	   = "FEDCBA9876543210FEDCBA9876543210";
    plainText =  "31323334353637383132333435363738313233343536373831323334353637383132333435363738313233343536373831323334353637383132333435363738";
    cipherText = "695347477477FC1E695347477477FC1E695347477477FC1E695347477477FC1E695347477477FC1E695347477477FC1E695347477477FC1E695347477477FC1E";
    mode 	   = kCCModeECB;
    alg		= kCCAlgorithmBlowfish;
    padding = ccNoPadding;
    
	plan_tests(kTestTestCount);
    
    retval = CCModeTestCase(keyStr, iv, mode, alg, padding, cipherText, plainText);
    
    ok(retval == 0, "Blowfish Test 1");
    accum += retval;
    return accum;
}
#endif
