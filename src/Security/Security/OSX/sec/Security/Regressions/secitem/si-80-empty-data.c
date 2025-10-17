/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 14, 2022.
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
#include <utilities/SecCFWrappers.h>
#include <stdlib.h>
#include <unistd.h>

#include "Security_regressions.h"

/* Test <rdar://problem/16669564>
   Retrieving attributes and data when item contains no data crashed.
 */

static void tests(void)
{
    CFDictionaryRef item = CFDictionaryCreateForCFTypes(NULL,
                                                        kSecClass, kSecClassGenericPassword,
                                                        kSecAttrAccount, CFSTR("empty-data-account-test"),
                                                        kSecAttrService, CFSTR("empty-data-svce-test"),
                                                        NULL);
    ok_status(SecItemAdd(item, NULL), "add generic password");

    CFDictionaryRef query = CFDictionaryCreateForCFTypes(NULL,
                                                         kSecClass, kSecClassGenericPassword,
                                                         kSecAttrService, CFSTR("empty-data-svce-test"),
                                                         kSecMatchLimit, kSecMatchLimitAll,
                                                         kSecReturnData, kCFBooleanTrue,
                                                         kSecReturnAttributes, kCFBooleanTrue,
                                                         NULL);
    CFTypeRef result;
    ok_status(SecItemCopyMatching(query, &result), "query generic password");
    ok(isArray(result) && CFArrayGetCount(result) == 1, "return 1-sized array of results");
    CFDictionaryRef row = CFArrayGetValueAtIndex(result, 0);
    ok(isDictionary(row), "array row is dictionary");
    ok(CFDictionaryGetValue(row, kSecValueData) == NULL, "result contains no data");
    ok(CFEqual(CFDictionaryGetValue(row, kSecAttrService), CFSTR("empty-data-svce-test")), "svce attribute is returned");
    ok(CFEqual(CFDictionaryGetValue(row, kSecAttrAccount), CFSTR("empty-data-account-test")), "account attribute is returned");

    CFRelease(result);
    CFRelease(query);
    query = CFDictionaryCreateForCFTypes(NULL,
                                         kSecClass, kSecClassGenericPassword,
                                         kSecAttrService, CFSTR("empty-data-svce-test"),
                                         NULL);
    ok_status(SecItemDelete(query), "delete testing item");

    CFRelease(query);
    CFRelease(item);
}

int si_80_empty_data(int argc, char *const *argv)
{
    plan_tests(8);

    tests();

    return 0;
}
