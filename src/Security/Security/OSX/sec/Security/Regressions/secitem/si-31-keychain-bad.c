/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 30, 2024.
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
#include "keychain/securityd/SecItemServer.h"

#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sqlite3.h>

#include "Security_regressions.h"

const uint8_t keychain_data[] = {
    0x62, 0x70, 0x6c, 0x69, 0x73, 0x74, 0x30, 0x30, 0xd2, 0x01, 0x02, 0x03,
    0x04, 0x5f, 0x10, 0x1b, 0x4e, 0x53, 0x57, 0x69, 0x6e, 0x64, 0x6f, 0x77,
    0x20, 0x46, 0x72, 0x61, 0x6d, 0x65, 0x20, 0x50, 0x72, 0x6f, 0x63, 0x65,
    0x73, 0x73, 0x50, 0x61, 0x6e, 0x65, 0x6c, 0x5f, 0x10, 0x1d, 0x4e, 0x53,
    0x57, 0x69, 0x6e, 0x64, 0x6f, 0x77, 0x20, 0x46, 0x72, 0x61, 0x6d, 0x65,
    0x20, 0x41, 0x62, 0x6f, 0x75, 0x74, 0x20, 0x54, 0x68, 0x69, 0x73, 0x20,
    0x4d, 0x61, 0x63, 0x5f, 0x10, 0x1c, 0x32, 0x38, 0x20, 0x33, 0x37, 0x33,
    0x20, 0x33, 0x34, 0x36, 0x20, 0x32, 0x39, 0x30, 0x20, 0x30, 0x20, 0x30,
    0x20, 0x31, 0x34, 0x34, 0x30, 0x20, 0x38, 0x37, 0x38, 0x20, 0x5f, 0x10,
    0x1d, 0x35, 0x36, 0x38, 0x20, 0x33, 0x39, 0x35, 0x20, 0x33, 0x30, 0x37,
    0x20, 0x33, 0x37, 0x39, 0x20, 0x30, 0x20, 0x30, 0x20, 0x31, 0x34, 0x34,
    0x30, 0x20, 0x38, 0x37, 0x38, 0x20, 0x08, 0x0d, 0x2b, 0x4b, 0x6a, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x05, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x8a
};

/* Test basic add delete update copy matching stuff. */
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
    int fd;
    ok_unix(fd = open(keychain_name, O_RDWR | O_CREAT | O_TRUNC, 0644),
        "create keychain file");
    is(write(fd, keychain_data, sizeof(keychain_data)),
        (ssize_t)sizeof(keychain_data), "write garbage to keychain file");
    ok_unix(close(fd), "close keychain file");

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

    CFRelease(query);
    CFRelease(eighty);
    CFRelease(pwdata);
#endif
}

int si_31_keychain_bad(int argc, char *const *argv)
{
	plan_tests(7);

	tests();

	return 0;
}
