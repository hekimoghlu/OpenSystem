/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 8, 2023.
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
//  CommonCryptorWithData.c
//  CommonCrypto
//
//  Created by Richard Murphy on 8/8/12.
//  Copyright (c) 2012 Platform Security. All rights reserved.
//

#include <stdio.h>
#include <CommonCrypto/CommonCryptor.h>
#include "testbyteBuffer.h"
#include "testmore.h"
#include "capabilities.h"

#if (CCWITHDATA == 0)
entryPoint(CommonCryptoWithData,"CommonCrypto With Data Testing")
#else
#define AES_KEYST_SIZE    (kCCContextSizeAES128 + 8)

static int kTestTestCount = 1;

int CommonCryptoWithData(int __unused argc, char *const * __unused argv)
{
    CCCryptorStatus retval;
    CCCryptorRef cryptor;
    unsigned char data[AES_KEYST_SIZE];

    
	plan_tests(kTestTestCount);
    
    byteBuffer key = hexStringToBytes("2b7e151628aed2a6abf7158809cf4f3c");
    retval = CCCryptorCreateFromData(kCCEncrypt, kCCAlgorithmAES128,
                                     kCCOptionECBMode, key->bytes, key->len, NULL,
                                     data, AES_KEYST_SIZE, &cryptor, NULL);

    CCCryptorRelease(cryptor);
    ok(retval == kCCSuccess, "Cryptor was created");
    free(key);
    return 0;
}
#endif
