/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 27, 2024.
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
#include "ccdebug.h"
#include <CommonCrypto/CommonCryptor.h>
#include <CommonCrypto/CommonCryptorSPI.h>

#include <corecrypto/cc.h>
#include <corecrypto/cc_priv.h>
#include <corecrypto/ccchacha20poly1305.h>
#include <corecrypto/ccchacha20poly1305_priv.h>

static inline CCCryptorStatus translate_corecrypto_error_code(int error)
{
    switch (error) {
        case CCERR_OK:
            return kCCSuccess;
        case CCERR_PARAMETER:
            return kCCParamError;
        default:
            return kCCUnspecifiedError;
    }
}

static inline CCCryptorStatus validate_parameters(size_t keyLength, size_t nonceLength)
{
    if (keyLength != CCCHACHA20_KEY_NBYTES) {
        return kCCKeySizeError;
    }
    if (nonceLength != CCCHACHA20_NONCE_NBYTES) {
        return kCCParamError;
    }

    return kCCSuccess;
}

CCCryptorStatus CCCryptorChaCha20(const void *key, size_t keyLength,
                                  const void *nonce, size_t nonceLength,
                                  uint32_t counter,
                                  const void *dataIn, size_t dataInLength, void *dataOut)
{
    CCCryptorStatus status = validate_parameters(keyLength, nonceLength);
    if (status != kCCSuccess) {
        return status;
    }

    int result = ccchacha20(key, nonce, counter, dataInLength, dataIn, dataOut);
    return translate_corecrypto_error_code(result);
}
