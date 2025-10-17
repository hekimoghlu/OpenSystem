/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 6, 2021.
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
//  si-82-token-ag.c
//  Copyright (c) 2013-2014 Apple Inc. All Rights Reserved.
//
//

#include <CoreFoundation/CoreFoundation.h>
#include <Security/SecItem.h>
#include <Security/SecItemPriv.h>
#include <Security/SecBase.h>
#include <utilities/array_size.h>
#include <utilities/SecCFWrappers.h>
#include <os/feature_private.h>

#include "Security_regressions.h"

static void tests(void) {
    CFMutableDictionaryRef dict = CFDictionaryCreateMutable(NULL, 0, NULL, NULL);
    CFDictionaryAddValue(dict, kSecClass, kSecClassGenericPassword);
    CFDictionaryAddValue(dict, kSecAttrService, CFSTR("test"));
    CFDictionaryAddValue(dict, kSecAttrAccessGroup, kSecAttrAccessGroupToken);

    is_status(SecItemAdd(dict, NULL), errSecMissingEntitlement);
    if (os_feature_enabled(CryptoTokenKit, UseTokens)) {
        is_status(SecItemCopyMatching(dict, NULL), errSecItemNotFound);
    } else {
        is_status(SecItemCopyMatching(dict, NULL), errSecMissingEntitlement);
    }

    CFRelease(dict);
}

int si_82_token_ag(int argc, char *const *argv) {

    plan_tests(2);
    tests();
    return 0;
}
