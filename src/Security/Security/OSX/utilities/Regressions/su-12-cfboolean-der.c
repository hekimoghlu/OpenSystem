/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 24, 2024.
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
#include "utilities/der_plist.h"
#include "utilities/der_plist_internal.h"

#include "utilities/SecCFRelease.h"
#include "utilities/array_size.h"

#include <CoreFoundation/CoreFoundation.h>

#include "utilities_regressions.h"

#define kMaxResultSize 16

#define kTestsPerTestCase 8
static void one_test(CFBooleanRef value, size_t der_size, const uint8_t *expected_der)
{

    uint8_t buffer[kMaxResultSize];
    uint8_t *buffer_end = buffer + sizeof(buffer);

    uint8_t* encoded = der_encode_plist(value, NULL, buffer, buffer_end);

    ok(encoded != NULL &&
       (der_size == (buffer_end - encoded)) &&
       (memcmp(encoded, expected_der, der_size) == 0));

    encoded = der_encode_boolean(value, NULL, buffer, buffer_end);

    ok(encoded != NULL &&
       (der_size == (buffer_end - encoded)) &&
       (memcmp(encoded, expected_der, der_size) == 0));

#if 0
    printf(".size = %d, .res = { ", (buffer_end - encoded));
    for(int c = 0; c < (buffer_end - encoded); ++c)
        printf("0x%02X, ", encoded[c]);
    printf("},\n");
#endif

    CFBooleanRef decoded = NULL;
    const uint8_t* decode_end = der_decode_boolean(NULL,
                                                   &decoded, NULL, encoded, buffer_end);

    ok(decode_end == buffer_end);
    ok((decoded != NULL) && CFEqual(decoded, value));

    CFPropertyListRef decoded_type = NULL;
    decode_end = der_decode_plist(NULL,
                                  &decoded_type, NULL, encoded, buffer_end);

    ok(decode_end == buffer_end);
    ok((decoded != NULL) && CFEqual(decoded_type, value));

    ok(der_sizeof_boolean(value, NULL) == der_size);
    ok(der_sizeof_plist(value, NULL) == der_size);

    CFReleaseNull(decoded);
    CFReleaseNull(decoded_type);
}

#define kTestCount (2 * kTestsPerTestCase)
static void tests(void)
{
    const uint8_t der_true[] = { 0x01, 0x01, 0x01 };

    one_test(kCFBooleanTrue, sizeof(der_true), der_true);
    
    const uint8_t der_false[] = { 0x01, 0x01, 0x00 };
    
    one_test(kCFBooleanFalse, sizeof(der_false), der_false);
}

int su_12_cfboolean_der(int argc, char *const *argv)
{
    plan_tests(kTestCount);
    tests();

    return 0;
}
