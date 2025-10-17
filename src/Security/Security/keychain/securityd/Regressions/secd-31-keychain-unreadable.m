/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 9, 2022.
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
#include <Security/SecBase.h>
#include <Security/SecItem.h>
#include <Security/SecInternal.h>
#include <utilities/SecFileLocations.h>
#include <utilities/SecCFWrappers.h>

#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sqlite3.h>

#include "secd_regressions.h"

#include "keychain/securityd/SecItemServer.h"

#include "SecdTestKeychainUtilities.h"


#if !(TARGET_OS_IOS && TARGET_OS_SIMULATOR)
static void setKeychainPermissions(int perm) {
    CFStringRef kc_path_cf = __SecKeychainCopyPath();
    CFStringPerformWithCString(kc_path_cf, ^(const char *path) {
        ok_unix(chmod(path, perm), "chmod keychain file %s to be %d", path, perm);
    });
}
#endif

int secd_31_keychain_unreadable(int argc, char *const *argv)
{
#if TARGET_OS_IOS && TARGET_OS_SIMULATOR
    // When running on iOS device in debugger, the target usually runs
    // as root, which means it has access to the file even after setting 000.
    return 0;
#else
    plan_tests(10 + kSecdTestSetupTestCount);
    secd_test_setup_temp_keychain("secd_31_keychain_unreadable", ^{
        CFStringRef keychain_path_cf = __SecKeychainCopyPath();
        
        CFStringPerformWithCString(keychain_path_cf, ^(const char *keychain_path) {
            int fd;
            ok_unix(fd = open(keychain_path, O_RDWR | O_CREAT | O_TRUNC, 0644),
                    "create keychain file '%s'", keychain_path);
            SKIP: {
                skip("Cannot fchmod keychain file with invalid fd", 2, fd > -1);
                ok_unix(fchmod(fd, 0), " keychain file '%s'", keychain_path);
                ok_unix(close(fd), "close keychain file '%s'", keychain_path);
            }
        });
        
        CFReleaseSafe(keychain_path_cf);
    });
    
    int v_eighty = 80;
    CFNumberRef eighty = CFNumberCreate(NULL, kCFNumberSInt32Type, &v_eighty);
    const char *v_data = "test";
    CFDataRef pwdata = CFDataCreate(NULL, (UInt8 *)v_data, strlen(v_data));
    CFMutableDictionaryRef query = CFDictionaryCreateMutable(NULL, 0, NULL, NULL);
    CFDictionaryAddValue(query, kSecClass, kSecClassInternetPassword);
    CFDictionaryAddValue(query, kSecAttrServer, CFSTR("members.spamcop.net"));
    CFDictionaryAddValue(query, kSecAttrAccount, CFSTR("smith"));
    CFDictionaryAddValue(query, kSecAttrPort, eighty);
    CFDictionaryAddValue(query, kSecAttrProtocol, kSecAttrProtocolHTTP);
    CFDictionaryAddValue(query, kSecAttrAuthenticationType, kSecAttrAuthenticationTypeDefault);
    CFDictionaryAddValue(query, kSecValueData, pwdata);
    
    is_status(SecItemAdd(query, NULL), errSecNotAvailable, "Cannot add items to unreadable keychain");
    is_status(SecItemCopyMatching(query, NULL), errSecNotAvailable, "Cannot read items in unreadable keychain");
    
    setKeychainPermissions(0644);
    
    ok_status(SecItemAdd(query, NULL), "Add internet password");
    is_status(SecItemAdd(query, NULL), errSecDuplicateItem,
              "Add internet password again");
    ok_status(SecItemCopyMatching(query, NULL), "Found the item we added");
    
    // For commented tests need to convince secd to let go of connections.
    // Without intervention it keeps them and accesses continue to succeed.
    /*
     setKeychainPermissions(0);
     is_status(SecItemCopyMatching(query, NULL), errSecNotAvailable, "Still cannot read items in unreadable keychain");
     
     setKeychainPermissions(0644);
     ok_status(SecItemCopyMatching(query, NULL), "Found the item again");
     */
    ok_status(SecItemDelete(query),"Deleted the item we added");
    
    CFReleaseNull(eighty);
    CFReleaseNull(pwdata);
    CFReleaseNull(query);

    secd_test_teardown_delete_temp_keychain("secd_31_keychain_unreadable");
#endif  // !(TARGET_OS_IOS && TARGET_OS_SIMULATOR)
    return 0;
}
