/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 3, 2022.
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
#include <Security/SecItem.h>
#include <Security/SecBase.h>
#include <utilities/array_size.h>
#include <stdlib.h>
#include <unistd.h>

#include "Security_regressions.h"

/* Test basic add delete update copy matching stuff. */
static void tests(void)
{
#ifndef NO_SERVER
    plan_skip_all("No testing against server.");
#else
    const void *keys[] = {
        kSecClass,
    };
    const void *values[] = {
        kSecClassInternetPassword,
    };
    CFDictionaryRef query = CFDictionaryCreate(NULL, keys, values,
    array_size(keys), NULL, NULL);
    CFTypeRef results = NULL;
    is_status(SecItemCopyMatching(query, &results), errSecItemNotFound,
    "find nothing");
    is(results, NULL, "results still NULL?");
    if (results) {
        CFRelease(results);
        results = NULL;
    }

    if (query) {
        CFRelease(query);
        query = NULL;
    }
#endif
}

int si_00_find_nothing(int argc, char *const *argv)
{
    plan_tests(2);
	tests();

	return 0;
}
