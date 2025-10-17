/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 28, 2024.
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
#include "keychain/securityd/SecItemServer.h"

#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sqlite3.h>

#include "Security_regressions.h"

#ifdef NO_SERVER
static void ensureKeychainExists(void) {
    CFDictionaryRef query = CFDictionaryCreate(0, (const void **)&kSecClass, (const void **)&kSecClassInternetPassword, 1, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
    CFTypeRef results = NULL;
    is_status(SecItemCopyMatching(query, &results), errSecItemNotFound, "expected nothing got %@", results);
    CFReleaseNull(query);
    CFReleaseNull(results);
}
#endif

/* Create an empty keychain file that can't be read or written and make sure
   securityd can deal with it. */
static void tests(void)
{
#ifndef NO_SERVER
    plan_skip_all("No testing against server.");
#else
    const char *home_dir = getenv("HOME");
    char keychain_dir[1000];
    char keychain_name[1000];
    snprintf(keychain_dir, sizeof(keychain_dir), "%s/Library/Keychains", home_dir);
    snprintf(keychain_name, sizeof(keychain_name), "%s/keychain-2-debug.db", keychain_dir);

    ensureKeychainExists();
    int fd;
    ok_unix(fd = open(keychain_name, O_RDWR | O_CREAT | O_TRUNC, 0644),
        "create keychain file '%s'", keychain_name);
    ok_unix(fchmod(fd, 0), " keychain file '%s'", keychain_name);
    ok_unix(close(fd), "close keychain file '%s'", keychain_name);

    SecKeychainDbReset(NULL);

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
    ok_status(SecItemAdd(query, NULL), "add internet password");
    is_status(SecItemAdd(query, NULL), errSecDuplicateItem,
	"add internet password again");

    ok_status(SecItemCopyMatching(query, NULL), "Found the item we added");

    ok_status(SecItemDelete(query),"Deleted the item we added");

    CFReleaseSafe(eighty);
    CFReleaseSafe(pwdata);
    CFReleaseSafe(query);
#endif
}

int si_31_keychain_unreadable(int argc, char *const *argv)
{
	plan_tests(8);
	tests();

	return 0;
}
