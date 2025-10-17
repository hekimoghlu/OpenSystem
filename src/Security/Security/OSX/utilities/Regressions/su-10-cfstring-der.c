/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 11, 2022.
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

#include <CoreFoundation/CFString.h>

#include "utilities_regressions.h"


#define kMaxResultSize 256

struct test_case {
    CFStringRef str;
    size_t      size;
    uint8_t     res[kMaxResultSize];
};

static struct test_case test_cases[] = {
    { .str = CFSTR("FOO"),                  .size = 5, .res = { 0x0C, 0x03, 0x46, 0x4F, 0x4F, } },
    { .str = CFSTR("!ÃŸâˆ‚Æ’Ë™Ã¥Å“Ë†Ã¸Â¥Â®xzfff"),     .size = 29, .res = { 0x0C, 0x1B, 0x21, 0xC3, 0x9F, 0xE2, 0x88, 0x82, 0xC6, 0x92,
                                                                 0xCB, 0x99, 0xC3, 0xA5, 0xC5, 0x93, 0xCB, 0x86, 0xC3, 0xB8,
                                                                 0xC2, 0xA5, 0xC2, 0xAE, 0x78, 0x7A, 0x66, 0x66, 0x66, } },
};


#define kTestsPerTestCase 8
static void one_test(const struct test_case * thisCase)
{
    uint8_t buffer[4 * kMaxResultSize + 1];
    uint8_t* buffer_end = buffer + sizeof(buffer);

    uint8_t* encoded = der_encode_string(thisCase->str, NULL, buffer, buffer_end);

    ok(encoded != NULL &&
       (thisCase->size == (buffer_end - encoded)) &&
       (memcmp(encoded, thisCase->res, thisCase->size) == 0));

    CFStringRef decoded = NULL;

    const uint8_t* decode_end = der_decode_string(NULL,
                                                  &decoded, NULL, encoded, buffer_end);

    ok(decode_end == buffer_end);
    ok((decoded != NULL) && CFEqual(decoded, thisCase->str));

    encoded = der_encode_plist(thisCase->str, NULL, buffer, buffer_end);

    ok(encoded != NULL &&
       (thisCase->size == (buffer_end - encoded)) &&
       (memcmp(encoded, thisCase->res, thisCase->size) == 0));

    CFTypeRef decoded_type = NULL;

    decode_end = der_decode_plist(NULL,
                                  &decoded_type, NULL, encoded, buffer_end);

    ok(decode_end == buffer_end);
    ok((decoded_type != NULL) && CFEqual(decoded_type, thisCase->str));

    ok(der_sizeof_string(thisCase->str, NULL) == thisCase->size);
    ok(der_sizeof_plist(thisCase->str, NULL) == thisCase->size);

    CFReleaseNull(decoded);
    CFReleaseNull(decoded_type);
}

#define kTestCount (array_size(test_cases) * kTestsPerTestCase)
static void tests(void)
{
    for (int testnumber = 0; testnumber < array_size(test_cases); ++testnumber)
        one_test(test_cases + testnumber);
}

int su_10_cfstring_der(int argc, char *const *argv)
{
    plan_tests(kTestCount);
    tests();

    return 0;
}
