/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 18, 2022.
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
#include "capabilities.h"
#include <stdio.h>
#include <CommonCrypto/CommonCryptor.h>
#include <CommonCrypto/CommonRandom.h>
#include <CommonCrypto/CommonCryptorSPI.h>
#include <Security/Security.h>
#include "CCCryptorTestFuncs.h"
#include "testbyteBuffer.h"
#include "testmore.h"
#include "CommonCryptorPriv.h"

#if (CCSYMCHACHA20POLY1305 == 0)
entryPoint(CommonCryptoSymChaCha20Poly1305, "CommonCrypto Symmetric ChaCha20Poly1305 Testing")
#else

static void testEncryptionInvalidParameters(void)
{
    uint8_t valid_key[32] = {};
    uint8_t valid_iv[12] = {};
    uint8_t valid_tag[16] = {};

    uint8_t plaintext[1024];
    (void)CCRandomGenerateBytes(plaintext, sizeof(plaintext));
    uint8_t ciphertext[sizeof(plaintext) + 16];

    // Invalid key length
    CCCryptorStatus status = CCCryptorChaCha20Poly1305OneshotEncrypt(valid_key, sizeof(valid_key) - 1, valid_iv, sizeof(valid_iv), NULL, 0, plaintext, sizeof(plaintext), ciphertext, valid_tag, sizeof(valid_tag));
    ok(status == kCCKeySizeError, "Expected %d, got %d", kCCKeySizeError, status);

    // Invalid IV length
    status = CCCryptorChaCha20Poly1305OneshotEncrypt(valid_key, sizeof(valid_key), valid_iv, sizeof(valid_iv) - 1, NULL, 0, plaintext, sizeof(plaintext), ciphertext, valid_tag, sizeof(valid_tag));
    ok(status == kCCParamError, "Expected %d, got %d", kCCParamError, status);

    // Invalid tag length
    status = CCCryptorChaCha20Poly1305OneshotEncrypt(valid_key, sizeof(valid_key), valid_iv, sizeof(valid_iv), NULL, 0, plaintext, sizeof(plaintext), ciphertext, valid_tag, sizeof(valid_tag) - 1);
    ok(status == kCCParamError, "Expected %d, got %d", kCCParamError, status);
}

#define CRYPTO_MAX_AAD_LENGTH 256
#define CRYPTO_MAX_PLAINTEXT_LENGTH 256

static void testEncryption_RoundTrip(void)
{
    uint8_t valid_key[32] = {};
    uint8_t valid_iv[12] = {};
    uint8_t valid_tag[16] = {};

    for (size_t aad_length = 0; aad_length < CRYPTO_MAX_AAD_LENGTH; aad_length++) {
        for (size_t plaintext_length = 1; plaintext_length < CRYPTO_MAX_PLAINTEXT_LENGTH; plaintext_length++) {
            uint8_t aad[CRYPTO_MAX_AAD_LENGTH] = {};
            uint8_t plaintext[CRYPTO_MAX_PLAINTEXT_LENGTH];
            uint8_t ciphertext[CRYPTO_MAX_PLAINTEXT_LENGTH];
            uint8_t recovered_plaintext[CRYPTO_MAX_PLAINTEXT_LENGTH];

            (void)CCRandomGenerateBytes(valid_key, sizeof(valid_key));
            (void)CCRandomGenerateBytes(valid_iv, sizeof(valid_iv));
            (void)CCRandomGenerateBytes(valid_tag, sizeof(valid_tag));
            (void)CCRandomGenerateBytes(aad, sizeof(aad));
            (void)CCRandomGenerateBytes(plaintext, sizeof(plaintext));

            CCCryptorStatus status = CCCryptorChaCha20Poly1305OneshotEncrypt(valid_key, sizeof(valid_key), valid_iv, sizeof(valid_iv), aad_length ? aad : NULL, aad_length, plaintext, plaintext_length, ciphertext, valid_tag, sizeof(valid_tag));
            ok(status == kCCSuccess, "Encryption failed. Expected %d, got %d", kCCSuccess, status);

            status = CCCryptorChaCha20Poly1305OneshotDecrypt(valid_key, sizeof(valid_key), valid_iv, sizeof(valid_iv), aad_length ? aad : NULL, aad_length, ciphertext, plaintext_length, recovered_plaintext, valid_tag, sizeof(valid_tag));
            ok(status == kCCSuccess, "Decryption failed. Expected %d, got %d", kCCSuccess, status);
            ok(0 == memcmp(plaintext, recovered_plaintext, plaintext_length), "Invalid plaintext");
        }
    }
}

static int kTestTestCount = 194731;

int
CommonCryptoSymChaCha20Poly1305(int __unused argc, char *const * __unused argv)
{
    plan_tests(kTestTestCount);
    testEncryptionInvalidParameters();
    testEncryption_RoundTrip();
    return 0;
}

#endif
