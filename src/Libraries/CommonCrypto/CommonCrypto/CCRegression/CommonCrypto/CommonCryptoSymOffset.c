/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 10, 2022.
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
#include <CommonCrypto/CommonCryptorSPI.h>
#include "testbyteBuffer.h"
#include "testmore.h"
#include "capabilities.h"

#if (CCSYMOFFSET == 0)
entryPoint(CommonCryptoSymOffset, "CommonCrypto Symmetric Unaligned Testing")
#else

#define BIG_BUFFER_NBYTES 4096

int CommonCryptoSymOffset(int __unused argc, char *const *__unused argv) {
    int accum = 0;
    uint8_t big_buffer_1[BIG_BUFFER_NBYTES] = {0};
    uint8_t big_buffer_2[BIG_BUFFER_NBYTES] = {0};
    uint8_t big_buffer_3[BIG_BUFFER_NBYTES] = {0};
    int i;
    size_t moved;
    CCCryptorStatus retval;
    byteBuffer key = hexStringToBytes("010203040506070809000a0b0c0d0e0f");

    int num_iterations = 5;
    plan_tests(num_iterations * 3);

    for (i = 0; i < num_iterations; i++) {
        retval = CCCrypt(kCCEncrypt, kCCAlgorithmAES128, 0, key->bytes, key->len,
                         NULL, big_buffer_1 + i, BIG_BUFFER_NBYTES - 16,
                         big_buffer_2 + i, BIG_BUFFER_NBYTES, &moved);
        ok(retval == 0, "Encrypt worked");
    
        retval = CCCrypt(kCCDecrypt, kCCAlgorithmAES128, 0, key->bytes, key->len,
                         NULL, big_buffer_2 + i, moved, big_buffer_3 + i,
                         BIG_BUFFER_NBYTES, &moved);
        ok(retval == 0, "Decrypt worked");
    
        if (moved != (BIG_BUFFER_NBYTES - 16))
            retval = 99;
        else if (memcmp(big_buffer_1 + i, big_buffer_3 + i, moved))
            retval = 999;
    
        ok(retval == 0, "Encrypt/Decrypt Cycle");
        accum += retval;
    }
    free(key);
    return accum != 0;
}
#endif
