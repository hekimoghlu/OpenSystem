/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 13, 2022.
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
//  testutil.c
//  CommonCrypto
//
//  Created by Richard Murphy on 1/21/14.
//  Copyright (c) 2014 Apple Inc. All rights reserved.
//

#include <stdio.h>
#include "testutil.h"

int expectedEqualsComputed(char *label, byteBuffer expected, byteBuffer computed) {
    int retval;
    
    if(expected == NULL) {
        printf("%s >>> Computed Results = \"%s\"\n", label, bytesToHexString(computed));
        retval = 1;
    } else if(!bytesAreEqual(expected, computed)) {
        diag(label);
        printByteBuffer(expected, "Expected: ");
        printByteBuffer(computed, "  Result: ");
        retval=0;
    } else {
        retval = 1;
    }
    return retval;
}



char *digestName(CCDigestAlgorithm digestSelector) {
    switch(digestSelector) {
        default: return "None";
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        case  kCCDigestMD2: return "MD2";
        case  kCCDigestMD4: return "MD4";
        case  kCCDigestMD5: return "MD5";
        case  kCCDigestRMD160: return "RMD160";
        case  kCCDigestSHA1: return "SHA1";
#pragma clang diagnostic pop
        case  kCCDigestSHA224: return "SHA224";
        case  kCCDigestSHA256: return "SHA256";
        case  kCCDigestSHA384: return "SHA384";
        case  kCCDigestSHA512: return "SHA512";
        case  kCCDigestSHA3_224: return "SHA3-224";
        case  kCCDigestSHA3_256: return "SHA3-256";
        case  kCCDigestSHA3_384: return "SHA3-384";
        case  kCCDigestSHA3_512: return "SHA3-512";
    }
}


CCHmacAlgorithm digestID2HMacID(CCDigestAlgorithm digestSelector) {
    CCHmacAlgorithm hmacAlg;
    switch(digestSelector) {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        case kCCDigestMD5: hmacAlg = kCCHmacAlgMD5; break;
        case kCCDigestSHA1: hmacAlg = kCCHmacAlgSHA1; break;
#pragma clang diagnostic pop
        case kCCDigestSHA224: hmacAlg = kCCHmacAlgSHA224; break;
        case kCCDigestSHA256: hmacAlg = kCCHmacAlgSHA256; break;
        case kCCDigestSHA384: hmacAlg = kCCHmacAlgSHA384; break;
        case kCCDigestSHA512: hmacAlg = kCCHmacAlgSHA512; break;
        default: return HMAC_UNIMP;
    }
    return hmacAlg;
}

CCPseudoRandomAlgorithm digestID2PRF(CCDigestAlgorithm digestSelector) {
    switch(digestSelector) {
        case 0: return 0;
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        case kCCDigestSHA1: return kCCPRFHmacAlgSHA1;
#pragma clang diagnostic pop
        case kCCDigestSHA224: return kCCPRFHmacAlgSHA224;
        case kCCDigestSHA256: return kCCPRFHmacAlgSHA256;
        case kCCDigestSHA384: return kCCPRFHmacAlgSHA384;
        case kCCDigestSHA512: return kCCPRFHmacAlgSHA512;
        default:
            diag("Unrecognized PRF translation for %s", digestName(digestSelector));
            return 0;
    }
}

