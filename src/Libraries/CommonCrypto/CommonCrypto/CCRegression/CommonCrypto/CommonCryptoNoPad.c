/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 2, 2022.
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
#include <CommonCrypto/CommonCryptor.h>
#include "testbyteBuffer.h"
#include "testmore.h"
#include "capabilities.h"

#if (CCNOPAD == 0)
entryPoint(CommonCryptoNoPad,"CommonCrypto NoPad Testing")
#else

static int kTestTestCount = 1152;

#define MAXSTART 64
#define MAXOUT 4096

static size_t testkeySize(CCAlgorithm alg) {
    switch(alg) {
        case kCCAlgorithmAES128: return kCCKeySizeAES128; // we're only testing nopad - we'll default to 128 bit keys
        case kCCAlgorithmDES: return kCCKeySizeDES;
        case kCCAlgorithm3DES: return kCCKeySize3DES;
        case kCCAlgorithmCAST: return kCCKeySizeMaxCAST;
        case kCCAlgorithmBlowfish: return kCCKeySizeMinBlowfish;
        default: return 16;
    }
}

static size_t testblockSize(CCAlgorithm alg) {
    switch(alg) {
        case kCCAlgorithmAES128: return kCCBlockSizeAES128; // we're only testing nopad - we'll default to 128 bit keys
        case kCCAlgorithmDES: return kCCBlockSizeDES;
        case kCCAlgorithm3DES: return kCCBlockSize3DES;
        case kCCAlgorithmCAST: return kCCBlockSizeCAST;
        case kCCAlgorithmBlowfish: return kCCBlockSizeBlowfish;
        default: return 16;
    }
}

static int
testFinal(CCOperation op, CCMode mode, CCAlgorithm alg,
          size_t inputLength)
{
    CCCryptorRef cryptorRef;
    CCCryptorStatus status;
    size_t keyLength = testkeySize(alg);
    size_t blockSize = testblockSize(alg);
    uint8_t iv[16];
    uint8_t key[32];
    uint8_t dataIn[MAXSTART] = {0};
    uint8_t dataOut[MAXOUT] = {0};
    size_t moved = 0;
    
    for(size_t i = 0; i < 16; i++) {
        iv[i]=0xa0;
        key[i]=0xcc; key[16+i] = 0xdd;
    }
    
    CCCryptorStatus expectedFinal = kCCSuccess;
    size_t expectedLen = (inputLength / blockSize) * blockSize;
    if(expectedLen != inputLength) expectedFinal = kCCAlignmentError;
    
    status = CCCryptorCreateWithMode(op, mode, alg, ccNoPadding, iv, key, keyLength, NULL, 0, 0, 0, &cryptorRef);
    ok(status == kCCSuccess, "Created Cryptor");

#if 0
    retval = CCCryptorGetOutputLength(cryptorRef, 0, true);
    ok(retval == expectedLen, "Got Length Value Expected");
    if(retval != expectedLen) {
        printf("inputLength = %lu Got %lu expected %lu\n", inputLength, retval, expectedLen);
    }
#endif
    
    status = CCCryptorUpdate(cryptorRef, dataIn, inputLength, dataOut, MAXOUT, &moved);
    ok(status == kCCSuccess, "Setup Initial Internal Length");
    
    status = CCCryptorFinal(cryptorRef, dataOut+moved, MAXOUT-moved, &moved);
    ok(status == expectedFinal, "Got Expected Final Result");
    if(status != expectedFinal) {
        printf("Expected Final Result %d got %d\n", expectedFinal, status);
    }
    status = CCCryptorRelease(cryptorRef);
    ok(status == kCCSuccess, "Released Cryptor");
    return -1;
}

int CommonCryptoNoPad (int __unused argc, char *const * __unused argv)
{
    int verbose = 0;
	plan_tests(kTestTestCount);
    
    CCAlgorithm algs[] = { kCCAlgorithmAES128, kCCAlgorithmDES, kCCAlgorithm3DES, kCCAlgorithmCAST, kCCAlgorithmBlowfish};
    CCAlgorithm alg;
    
    /* ENCRYPTING */
    for(size_t algnum = 0; algnum < sizeof(algs)/sizeof(CCAlgorithm); algnum++) {
        alg = algs[algnum];
        if(verbose) diag("ENCRYPTING AES-CBC-NoPadding Update");
        for(size_t i=0; i<3*testblockSize(alg); i++) {
            testFinal(kCCEncrypt, kCCModeCBC, alg, i);
        }
    }
    
    /* DECRYPTING */
    for(size_t algnum = 0; algnum < sizeof(algs)/sizeof(CCAlgorithm); algnum++) {
        alg = algs[algnum];
        if(verbose) diag("DECRYPTING AES-CBC-NoPadding Update");
        for(size_t i=0; i<3*testblockSize(alg); i++) {
            testFinal(kCCDecrypt, kCCModeCBC, alg, i);
        }
    }
    return 0;
}
#endif /* CCNOPAD */

