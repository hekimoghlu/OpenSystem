/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 8, 2022.
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

static void dumpSearchList(char * label, CFArrayRef searchList) {
    printf("%s:\n", label);

    for(int i = 0; i < CFArrayGetCount(searchList); i++) {
        char pathName[300];
        UInt32 len = sizeof(pathName);

        SecKeychainGetPath((SecKeychainRef) CFArrayGetValueAtIndex(searchList, i), &len, pathName);
        printf("   %s\n", pathName);
    }
    printf("\n");
}

static CFComparisonResult compare(const void* first, const void* second, void* context) {
    SecKeychainRef k1 = (SecKeychainRef) first;
    SecKeychainRef k2 = (SecKeychainRef) second;

    char path1[200];
    char path2[200];
    UInt32 l1 = 200, l2 = 200;

    SecKeychainGetPath(k1, &l1, path1);
    SecKeychainGetPath(k2, &l2, path2);

    return strcmp(path1, path2);
}

// Checks that these lists are equal modulo order
static bool keychainListsEqual(CFArrayRef list1, CFArrayRef list2) {

    CFIndex size1 = CFArrayGetCount(list1);
    CFIndex size2 = CFArrayGetCount(list2);

    if(size1 != size2) {
        return false;
    }

    CFMutableArrayRef m1 = CFArrayCreateMutableCopy(NULL, 0, list1);
    CFMutableArrayRef m2 = CFArrayCreateMutableCopy(NULL, 0, list2);

    CFArraySortValues(m1, CFRangeMake(0, size1), &compare, NULL);
    CFArraySortValues(m2, CFRangeMake(0, size2), &compare, NULL);

    bool result = CFEqual(m1, m2);

    CFRelease(m1);
    CFRelease(m2);

    return result;
}

static void tests(void)
{
    SecKeychainRef kc = getPopulatedTestKeychain();

    CFArrayRef searchList = NULL;
    ok_status(SecKeychainCopySearchList(&searchList), "%s: SecKeychainCopySearchList", testName);
    dumpSearchList("initial", searchList);

    CFMutableArrayRef mutableSearchList = CFArrayCreateMutableCopy(NULL, CFArrayGetCount(searchList) + 1, searchList);
    CFArrayAppendValue(mutableSearchList, kc);
    ok_status(SecKeychainSetSearchList(mutableSearchList), "%s: SecKeychainSetSearchList", testName);
    dumpSearchList("to set", mutableSearchList);

    CFArrayRef midSearchList = NULL;
    ok_status(SecKeychainCopySearchList(&midSearchList), "%s: SecKeychainCopySearchList (mid)", testName);
    dumpSearchList("after set", midSearchList);

    ok(keychainListsEqual(mutableSearchList, midSearchList), "%s: retrieved search list equal to set search list", testName);

    ok_status(SecKeychainDelete(kc), "%s: SecKeychainDelete", testName);
    CFReleaseNull(kc);

    CFArrayRef finalSearchList = NULL;
    ok_status(SecKeychainCopySearchList(&finalSearchList), "%s: SecKeychainCopySearchList (final)", testName);
    dumpSearchList("final", finalSearchList);

    ok(keychainListsEqual(finalSearchList, searchList), "%s: final search list equal to initial search list", testName);

    CFRelease(searchList);
    CFRelease(mutableSearchList);
    CFRelease(midSearchList);
    CFRelease(finalSearchList);
}

int kc_03_keychain_list(int argc, char *const *argv)
{
    plan_tests(9);
    initializeKeychainTests(__FUNCTION__);

    tests();

    deleteTestFiles();
    return 0;
}
