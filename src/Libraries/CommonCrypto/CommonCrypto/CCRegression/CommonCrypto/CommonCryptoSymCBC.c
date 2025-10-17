/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 5, 2022.
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
 *  CBCTest.c
 *  CommonCrypto
 */

#include <stdio.h>
#include "CCCryptorTestFuncs.h"
#include "testbyteBuffer.h"
#include "testmore.h"
#include "capabilities.h"

#if (CCSYMCBC == 0)
entryPoint(CommonCryptoSymCBC,"CommonCrypto Symmetric CBC Testing")
#else


static int kTestTestCount = 31;

int CommonCryptoSymCBC(int __unused argc, char *const * __unused argv) {
	char *keyStr;
	char *iv;
	char *plainText;
	char *cipherText;
    CCAlgorithm alg;
    CCOptions options;
	int retval;
    int accum = 0;

	keyStr 	   = "000102030405060708090a0b0c0d0e0f";
	iv         = "0f0e0d0c0b0a09080706050403020100";
    alg		   = kCCAlgorithmAES128;
    options    = kCCOptionPKCS7Padding;
    
	plan_tests(kTestTestCount);
    
    accum = (int) genRandomSize(1,10);

	// 1
	plainText  = "0a";
	cipherText = "a385b047a4108a8748bf96b435738213";
    retval = CCCryptTestCase(keyStr, iv, alg, options, cipherText, plainText, true);
    ok(retval == 0, "CBC with Padding 1 byte CCCrypt");
    accum |= retval;
    retval = CCMultiCryptTestCase(keyStr, iv, alg, options, cipherText, plainText);
    ok(retval == 0, "CBC with Padding 1 byte Multiple Updates");
    accum |= retval;

	// 15
	plainText  = "0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a";
	cipherText = "324a44cf3395b14214861084019f9257";
    retval = CCCryptTestCase(keyStr, iv, alg, options, cipherText, plainText, true);
    ok(retval == 0, "CBC with Padding 15 byte CCCrypt");
    accum |= retval;
    retval = CCMultiCryptTestCase(keyStr, iv, alg, options, cipherText, plainText);
    ok(retval == 0, "CBC with Padding 15 byte Multiple Updates");
    accum |= retval;

	// 16
    plainText  = "0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a";
	cipherText = "16d67a52c1e8384f7ed887c2011605346544febcf84574c334f1145d17567047";
    retval = CCCryptTestCase(keyStr, iv, alg, options, cipherText, plainText, true);
    ok(retval == 0, "CBC with Padding 16 byte CCCrypt");
    accum |= retval;
    retval = CCMultiCryptTestCase(keyStr, iv, alg, options, cipherText, plainText);
    ok(retval == 0, "CBC with Padding 16 byte Multiple Updates");
    accum |= retval;

	// 17
	plainText  = "0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a";
	cipherText = "16d67a52c1e8384f7ed887c2011605348b72cecb00bbc00f328af6bb69085b02";
    retval = CCCryptTestCase(keyStr, iv, alg, options, cipherText, plainText, true);
    ok(retval == 0, "CBC with Padding 17 byte CCCrypt");
    accum |= retval;
    retval = CCMultiCryptTestCase(keyStr, iv, alg, options, cipherText, plainText);
    ok(retval == 0, "CBC with Padding 17 byte Multiple Updates");
    accum |= retval;

	// 31
	plainText  = "0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a";
	cipherText = "16d67a52c1e8384f7ed887c2011605347175cf878a75bc1947ae79c6c6835030";
    retval = CCCryptTestCase(keyStr, iv, alg, options, cipherText, plainText, true);
    ok(retval == 0, "CBC with Padding 31 byte CCCrypt");
    accum |= retval;
    retval = CCMultiCryptTestCase(keyStr, iv, alg, options, cipherText, plainText);
    ok(retval == 0, "CBC with Padding 31 byte Multiple Updates");
    accum |= retval;

	// 32
	plainText  = "0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a";
	cipherText = "16d67a52c1e8384f7ed887c20116053486869f3b83f3b3a83531e4169e97b7244a49199daa033fa88f07dd4be52ae78e";
    retval = CCCryptTestCase(keyStr, iv, alg, options, cipherText, plainText, true);
    ok(retval == 0, "CBC with Padding 32 byte CCCrypt");
    accum |= retval;
    retval = CCMultiCryptTestCase(keyStr, iv, alg, options, cipherText, plainText);
    ok(retval == 0, "CBC with Padding 32 byte Multiple Updates");
    accum |= retval;

	// 33
	plainText  = "0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a";
	cipherText = "16d67a52c1e8384f7ed887c20116053486869f3b83f3b3a83531e4169e97b724d0080fb874dd556fa86b314acc4f597b";
    retval = CCCryptTestCase(keyStr, iv, alg, options, cipherText, plainText, true);
    ok(retval == 0, "CBC with Padding 33 byte CCCrypt");
    accum |= retval;
    retval = CCMultiCryptTestCase(keyStr, iv, alg, options, cipherText, plainText);
    ok(retval == 0, "CBC with Padding 33 byte Multiple Updates");
    accum |= retval;
    
    iv = NULL;
	// 1
	plainText  = "0a";
	cipherText = "27cae51ac763b250945fd805c937119b";
    retval = CCCryptTestCase(keyStr, iv, alg, options, cipherText, plainText, true);
    ok(retval == 0, "CBC with Padding 1 byte CCCrypt NULL IV");
    accum |= retval;
    retval = CCMultiCryptTestCase(keyStr, iv, alg, options, cipherText, plainText);
    ok(retval == 0, "CBC with Padding 1 byte Multiple Updates NULL IV");
    accum |= retval;

	// 15
	plainText  = "0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a";
	cipherText = "feb9c3a005dcbd1e2630af742e988e81";
    retval = CCCryptTestCase(keyStr, iv, alg, options, cipherText, plainText, true);
    ok(retval == 0, "CBC with Padding 15 byte CCCrypt NULL IV");
    accum |= retval;
    retval = CCMultiCryptTestCase(keyStr, iv, alg, options, cipherText, plainText);
    ok(retval == 0, "CBC with Padding 15 byte Multiple Updates NULL IV");
    accum |= retval;

	// 16
    plainText  = "0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a";
	cipherText = "d307b25d3abaf87c0053e8188152992a8b002a94911ee1e157d815a026cfadeb";
    retval = CCCryptTestCase(keyStr, iv, alg, options, cipherText, plainText, true);
    ok(retval == 0, "CBC with Padding 16 byte CCCrypt NULL IV");
    retval = CCMultiCryptTestCase(keyStr, iv, alg, options, cipherText, plainText);
    ok(retval == 0, "CBC with Padding 16 byte Multiple Updates NULL IV");
    accum |= retval;

	// 17
	plainText  = "0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a";
	cipherText = "d307b25d3abaf87c0053e8188152992ab8fe4130b613e93617b2eda2e0c5c678";
    retval = CCCryptTestCase(keyStr, iv, alg, options, cipherText, plainText, true);
    ok(retval == 0, "CBC with Padding 17 byte CCCrypt NULL IV");
    accum |= retval;
    retval = CCMultiCryptTestCase(keyStr, iv, alg, options, cipherText, plainText);
    ok(retval == 0, "CBC with Padding 17 byte Multiple Updates NULL IV");
    accum |= retval;

	// 31
	plainText  = "0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a";
	cipherText = "d307b25d3abaf87c0053e8188152992a4157ad665141a79481f463357707f759";
    retval = CCCryptTestCase(keyStr, iv, alg, options, cipherText, plainText, true);
    ok(retval == 0, "CBC with Padding 31 byte CCCrypt NULL IV");
    accum |= retval;
    retval = CCMultiCryptTestCase(keyStr, iv, alg, options, cipherText, plainText);
    ok(retval == 0, "CBC with Padding 31 byte Multiple Updates NULL IV");
    accum |= retval;

	// 32
	plainText  = "0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a";
	cipherText = "d307b25d3abaf87c0053e8188152992a923832530aa268661a6c1fa3c69d6a23dc6d5c0d7fa8127cfd601cae71b4c14f";
    retval = CCCryptTestCase(keyStr, iv, alg, options, cipherText, plainText, true);
    ok(retval == 0, "CBC with Padding 32 byte CCCrypt NULL IV");
    accum |= retval;
    retval = CCMultiCryptTestCase(keyStr, iv, alg, options, cipherText, plainText);
    ok(retval == 0, "CBC with Padding 32 byte Multiple Updates NULL IV");
    accum |= retval;

	// 33
	plainText  = "0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a";
	cipherText = "d307b25d3abaf87c0053e8188152992a923832530aa268661a6c1fa3c69d6a2382178b537aa2946f7a4124ee33744edd";
    retval = CCCryptTestCase(keyStr, iv, alg, options, cipherText, plainText, true);
    ok(retval == 0, "CBC with Padding 33 byte CCCrypt NULL IV");
    accum |= retval;
    retval = CCMultiCryptTestCase(keyStr, iv, alg, options, cipherText, plainText);
    ok(retval == 0, "CBC with Padding 33 byte Multiple Updates NULL IV");
    accum |= retval;

    // 34 case test 1 repeated with wrong key size - negative test - don't let CCCryptTestCase() to print error messages on the console
    char keyStr_incorrect[strlen(keyStr)+1+2];
    strlcpy(keyStr_incorrect, keyStr, sizeof(keyStr_incorrect));
    strlcat(keyStr_incorrect, "01", sizeof(keyStr_incorrect));
    plainText  = "0a";
    cipherText = "a385b047a4108a8748bf96b435738213";
    retval = CCCryptTestCase(keyStr_incorrect, iv, alg, options, cipherText, plainText, false);
    ok(retval != 0, "CBC with wrong key size");

    // Blowfish vector that was failing for Jim
    
    alg = kCCAlgorithmBlowfish;
    options = 0;
    keyStr = "0123456789ABCDEFF0E1D2C3B4A59687";
    iv = "FEDCBA9876543210";
    plainText =  "37363534333231204E6F77206973207468652074696D6520666F722000000000";
    cipherText = "6B77B4D63006DEE605B156E27403979358DEB9E7154616D959F1652BD5FF92CC";
    retval = CCCryptTestCase(keyStr, iv, alg, options, cipherText, plainText, true);
    ok(retval == 0, "CBC-blowfish vector 1");
    accum |= retval;
    retval = CCMultiCryptTestCase(keyStr, iv, alg, options, cipherText, plainText);
    ok(retval == 0, "CBC-blowfish vector 1");
    accum |= retval;

    
    return accum != 0;
}
#endif

