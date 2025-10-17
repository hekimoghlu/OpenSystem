/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 30, 2025.
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
#include <CoreFoundation/CoreFoundation.h>
#include <Security/SecInternal.h>
#include <Security/SecItem.h>
#include <Security/SecBase.h>
#include <CommonCrypto/CommonHMAC.h>
#include <stdlib.h>
#include <unistd.h>
#include <Security/SecPBKDF.h>

#include "Security_regressions.h"

static int kTestTestCount = 16;

static void tests(void)
{
    {
        const char *password =          "password";
        const char *salt =              "salt";
        const int iterations =          1;
        const uint8_t expected[20] =  { 0x0c, 0x60, 0xc8, 0x0f,
                                        0x96, 0x1f, 0x0e, 0x71,
                                        0xf3, 0xa9, 0xb5, 0x24,
                                        0xaf, 0x60, 0x12, 0x06,
                                        0x2f, 0xe0, 0x37, 0xa6 };

        const char resultSize = sizeof(expected);

        uint8_t actual[resultSize];

        is(pbkdf2_hmac_sha1((const uint8_t*) password, strlen(password), (const uint8_t*) salt, strlen(salt), iterations, actual, resultSize), errSecSuccess, "pbkdf-sha-1: Failed Key Derivation I-1");

        is(memcmp(expected, actual, resultSize), 0, "pbkdf-sha-1: P-'password' S-'salt' I-1");
    }

    {
        const char *password =          "password";
        const char *salt =              "salt";
        const int iterations =          2;
        const uint8_t expected[20] =  { 0xea, 0x6c, 0x01, 0x4d,
                                        0xc7, 0x2d, 0x6f, 0x8c,
                                        0xcd, 0x1e, 0xd9, 0x2a,
                                        0xce, 0x1d, 0x41, 0xf0,
                                        0xd8, 0xde, 0x89, 0x57 };

        const char resultSize = sizeof(expected);

        uint8_t actual[resultSize];

        is(pbkdf2_hmac_sha1((const uint8_t*) password, strlen(password), (const uint8_t*) salt, strlen(salt), iterations, actual, resultSize), errSecSuccess, "pbkdf-sha-1: Failed Key Derivation I-2");

        is(memcmp(expected, actual, resultSize), 0, "pbkdf-sha-1: P-'password' S-'salt' I-2");
    }

    {
        const char *password =          "password";
        const char *salt =              "salt";
        const int iterations =          4096;
        const uint8_t expected[20] =  { 0x4b, 0x00, 0x79, 0x01,
                                        0xb7, 0x65, 0x48, 0x9a,
                                        0xbe, 0xad, 0x49, 0xd9,
                                        0x26, 0xf7, 0x21, 0xd0,
                                        0x65, 0xa4, 0x29, 0xc1 };

        const char resultSize = sizeof(expected);

        uint8_t actual[resultSize];

        is(pbkdf2_hmac_sha1((const uint8_t*) password, strlen(password), (const uint8_t*) salt, strlen(salt), iterations, actual, resultSize), errSecSuccess, "pbkdf-sha-1: Failed Key Derivation I-4096");

        is(memcmp(expected, actual, resultSize), 0, "pbkdf-sha-1: P-'password' S-'salt' I-4096");
    }

    SKIP: {
        skip("16777216 iterations is too slow", 1, 0);

        const char *password =          "password";
        const char *salt =              "salt";
        const int iterations =          16777216;
        const uint8_t expected[20] =  { 0xee, 0xfe, 0x3d, 0x61,
                                        0xcd, 0x4d, 0xa4, 0xe4,
                                        0xe9, 0x94, 0x5b, 0x3d,
                                        0x6b, 0xa2, 0x15, 0x8c,
                                        0x26, 0x34, 0xe9, 0x84 };

        const char resultSize = sizeof(expected);

        uint8_t actual[resultSize];

        is(pbkdf2_hmac_sha1((const uint8_t*) password, strlen(password), (const uint8_t*) salt, strlen(salt), iterations, actual, resultSize), errSecSuccess, "pbkdf-sha-1: Failed Key Derivation I-16777216");

        is(memcmp(expected, actual, resultSize), 0, "pbkdf-sha-1: P-'password' S-'salt' I-16777216");
    }


    {
        CFStringRef password    = CFStringCreateWithCString(NULL, "password", kCFStringEncodingUTF8);
        CFStringRef salt        = CFStringCreateWithCString(NULL, "salt", kCFStringEncodingUTF8);

        CFDataRef   passwordData    = CFStringCreateExternalRepresentation(NULL, password, kCFStringEncodingUTF8, 0);
        CFDataRef   saltData        = CFStringCreateExternalRepresentation(NULL, salt, kCFStringEncodingUTF8, 0);

        const int iterations =          1;
        const uint8_t expected[20] =  { 0x0c, 0x60, 0xc8, 0x0f,
                                        0x96, 0x1f, 0x0e, 0x71,
                                        0xf3, 0xa9, 0xb5, 0x24,
                                        0xaf, 0x60, 0x12, 0x06,
                                        0x2f, 0xe0, 0x37, 0xa6 };

        const char resultSize = sizeof(expected);

        CFMutableDataRef resultData = CFDataCreateMutable(NULL, resultSize);
        CFDataIncreaseLength(resultData, resultSize);

        is(SecKeyFromPassphraseDataHMACSHA1(passwordData, saltData, iterations, resultData), errSecSuccess, "pbkdf-sha-1: Failed Key Derivation I-1");

        is(memcmp(expected, CFDataGetBytePtr(resultData), resultSize), 0, "pbkdf-sha-1: P-'password' S-'salt' I-1");

        CFReleaseSafe(password);
        CFReleaseSafe(salt);
        CFReleaseSafe(passwordData);
        CFReleaseSafe(saltData);
        CFReleaseSafe(resultData);
    }

    {
        CFStringRef password    = CFStringCreateWithCString(NULL, "password", kCFStringEncodingUTF8);
        CFStringRef salt        = CFStringCreateWithCString(NULL, "salt", kCFStringEncodingUTF8);

        CFDataRef   passwordData    = CFStringCreateExternalRepresentation(NULL, password, kCFStringEncodingUTF8, 0);
        CFDataRef   saltData        = CFStringCreateExternalRepresentation(NULL, salt, kCFStringEncodingUTF8, 0);

        const int iterations =          2;
        const uint8_t expected[20] =  { 0xea, 0x6c, 0x01, 0x4d,
                                        0xc7, 0x2d, 0x6f, 0x8c,
                                        0xcd, 0x1e, 0xd9, 0x2a,
                                        0xce, 0x1d, 0x41, 0xf0,
                                        0xd8, 0xde, 0x89, 0x57 };

        const char resultSize = sizeof(expected);

        CFMutableDataRef resultData = CFDataCreateMutable(NULL, resultSize);
        CFDataIncreaseLength(resultData, resultSize);

        is(SecKeyFromPassphraseDataHMACSHA1(passwordData, saltData, iterations, resultData), errSecSuccess, "pbkdf-sha-1: Failed Key Derivation I-2");

        is(memcmp(expected, CFDataGetBytePtr(resultData), resultSize), 0, "pbkdf-sha-1: P-'password' S-'salt' I-2");

        CFReleaseSafe(password);
        CFReleaseSafe(salt);
        CFReleaseSafe(passwordData);
        CFReleaseSafe(saltData);
        CFReleaseSafe(resultData);
    }

    {
        CFStringRef password    = CFStringCreateWithCString(NULL, "password", kCFStringEncodingUTF8);
        CFStringRef salt        = CFStringCreateWithCString(NULL, "salt", kCFStringEncodingUTF8);

        CFDataRef   passwordData    = CFStringCreateExternalRepresentation(NULL, password, kCFStringEncodingUTF8, 0);
        CFDataRef   saltData        = CFStringCreateExternalRepresentation(NULL, salt, kCFStringEncodingUTF8, 0);

        const int iterations =          4096;
        const uint8_t expected[20] =  { 0x4b, 0x00, 0x79, 0x01,
                                        0xb7, 0x65, 0x48, 0x9a,
                                        0xbe, 0xad, 0x49, 0xd9,
                                        0x26, 0xf7, 0x21, 0xd0,
                                        0x65, 0xa4, 0x29, 0xc1 };


        const char resultSize = sizeof(expected);

        CFMutableDataRef resultData = CFDataCreateMutable(NULL, resultSize);
        CFDataIncreaseLength(resultData, resultSize);

        is(SecKeyFromPassphraseDataHMACSHA1(passwordData, saltData, iterations, resultData), errSecSuccess, "pbkdf-sha-1: Failed Key Derivation I-4096");

        is(memcmp(expected, CFDataGetBytePtr(resultData), resultSize), 0, "pbkdf-sha-1: P-'password' S-'salt' I-4096");

        CFReleaseSafe(password);
        CFReleaseSafe(salt);
        CFReleaseSafe(passwordData);
        CFReleaseSafe(saltData);
        CFReleaseSafe(resultData);
    }

    SKIP: {
        skip("16777216 iterations is too slow", 2, 0);

        CFStringRef password    = CFStringCreateWithCString(NULL, "password", kCFStringEncodingUTF8);
        CFStringRef salt        = CFStringCreateWithCString(NULL, "salt", kCFStringEncodingUTF8);

        CFDataRef   passwordData    = CFStringCreateExternalRepresentation(NULL, password, kCFStringEncodingUTF8, 0);
        CFDataRef   saltData        = CFStringCreateExternalRepresentation(NULL, salt, kCFStringEncodingUTF8, 0);

        const int iterations =          16777216;
        const uint8_t expected[20] =  { 0xee, 0xfe, 0x3d, 0x61,
                                        0xcd, 0x4d, 0xa4, 0xe4,
                                        0xe9, 0x94, 0x5b, 0x3d,
                                        0x6b, 0xa2, 0x15, 0x8c,
                                        0x26, 0x34, 0xe9, 0x84 };


        const char resultSize = sizeof(expected);

        CFMutableDataRef resultData = CFDataCreateMutable(NULL, resultSize);
        CFDataIncreaseLength(resultData, resultSize);

        is(SecKeyFromPassphraseDataHMACSHA1(passwordData, saltData, iterations, resultData), errSecSuccess, "pbkdf-sha-1: Failed Key Derivation I-16777216");

        is(memcmp(expected, CFDataGetBytePtr(resultData), resultSize), 0, "pbkdf-sha-1: P-'password' S-'salt' I-16777216");

        CFReleaseSafe(password);
        CFReleaseSafe(salt);
        CFReleaseSafe(passwordData);
        CFReleaseSafe(saltData);
        CFReleaseSafe(resultData);
    }

}

int spbkdf_00_hmac_sha1(int argc, char *const *argv)
{
	plan_tests(kTestTestCount);

	tests();

	return 0;
}
