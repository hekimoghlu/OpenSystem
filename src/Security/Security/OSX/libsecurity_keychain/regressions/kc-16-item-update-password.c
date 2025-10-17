/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 18, 2022.
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
#import <Security/Security.h>
#import <Security/SecCertificatePriv.h>

#include "keychain_regressions.h"
#include "kc-helpers.h"
#include "kc-item-helpers.h"

static void tests(void)
{
    SecKeychainRef kc = getPopulatedTestKeychain();

    CFMutableDictionaryRef query = NULL;
    SecKeychainItemRef item = NULL;

    // Find passwords
    query = createQueryCustomItemDictionaryWithService(kc, kSecClassInternetPassword, CFSTR("test_service"), CFSTR("test_service"));
    item = checkNCopyFirst(testName, query, 1);
    readPasswordContents(item, CFSTR("test_password"));  checkPrompts(0, "after reading a password");
    changePasswordContents(item, CFSTR("new_password")); checkPrompts(0, "changing a internet password");
    readPasswordContents(item, CFSTR("new_password"));   checkPrompts(0, "reading a changed internet password");
    CFReleaseNull(item);

    query = createQueryCustomItemDictionaryWithService(kc, kSecClassInternetPassword, CFSTR("test_service_restrictive_acl"), CFSTR("test_service_restrictive_acl"));
    item = checkNCopyFirst(testName, query, 1);
    readPasswordContentsWithResult(item, errSecAuthFailed, NULL); // we don't expect to be able to read this
    checkPrompts(1, "trying to read internet password without access");

    changePasswordContents(item, CFSTR("new_password"));
    checkPrompts(0, "after changing a internet password without access"); // NOTE: we expect this write to succeed, even though we're not on the ACL. Therefore, we should see 0 prompts for this step.
    readPasswordContentsWithResult(item, errSecAuthFailed, NULL); // we don't expect to be able to read this
    checkPrompts(1, "after changing a internet password without access");
    CFReleaseNull(item);

    query = createQueryCustomItemDictionaryWithService(kc, kSecClassGenericPassword, CFSTR("test_service"), CFSTR("test_service"));
    item = checkNCopyFirst(testName, query, 1);
    readPasswordContents(item, CFSTR("test_password"));   checkPrompts(0, "after reading a generic password");
    changePasswordContents(item, CFSTR("new_password"));  checkPrompts(0, "changing a generic password");
    readPasswordContents(item, CFSTR("new_password"));    checkPrompts(0, "after changing a generic password");
    CFReleaseNull(item);

    query = createQueryCustomItemDictionaryWithService(kc, kSecClassGenericPassword, CFSTR("test_service_restrictive_acl"), CFSTR("test_service_restrictive_acl"));
    item = checkNCopyFirst(testName, query, 1);
    readPasswordContentsWithResult(item, errSecAuthFailed, NULL); // we don't expect to be able to read this
    checkPrompts(1, "trying to read generic password without access");

    changePasswordContents(item, CFSTR("new_password"));
    checkPrompts(0, "changing a generic password without access"); // NOTE: we expect this write to succeed, even though we're not on the ACL. Therefore, we should see 0 prompts for this step.
    readPasswordContentsWithResult(item, errSecAuthFailed, NULL); // we don't expect to be able to read this
    checkPrompts(1, "after changing a generic password without access");
    CFReleaseNull(item);

    ok_status(SecKeychainDelete(kc), "%s: SecKeychainDelete", testName);
    CFReleaseNull(kc);
}
#define numTests (getPopulatedTestKeychainTests + \
checkNTests + readPasswordContentsTests + checkPromptsTests + changePasswordContentsTests + checkPromptsTests + readPasswordContentsTests + checkPromptsTests + \
checkNTests + readPasswordContentsTests + checkPromptsTests + changePasswordContentsTests + checkPromptsTests + readPasswordContentsTests + checkPromptsTests + \
checkNTests + readPasswordContentsTests + checkPromptsTests + changePasswordContentsTests + checkPromptsTests + readPasswordContentsTests + checkPromptsTests + \
checkNTests + readPasswordContentsTests + checkPromptsTests + changePasswordContentsTests + checkPromptsTests + readPasswordContentsTests + checkPromptsTests + \
+ 1)

int kc_16_item_update_password(int argc, char *const *argv)
{
    plan_tests(numTests);
    initializeKeychainTests(__FUNCTION__);

    tests();

    deleteTestFiles();
    return 0;
}
