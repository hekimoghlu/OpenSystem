/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 7, 2025.
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
#import <Foundation/Foundation.h>
#import <CoreFoundation/CoreFoundation.h>
#import <Security/SecBase.h>
#import <Security/SecItem.h>
#import <Security/SecItemPriv.h>
#import <Security/SecInternal.h>
#import <utilities/SecCFRelease.h>
#import <utilities/SecFileLocations.h>
#import "keychain/securityd/SecItemServer.h"

#import <stdlib.h>

#include "secd_regressions.h"
#include "SecdTestKeychainUtilities.h"
#include "server_security_helpers.h"

static int ckmirror_row_exists = 0;
static int ckmirror_row_callback(void* unused, int count, char **data, char **columns)
{
    ckmirror_row_exists = 1;
    for (int i = 0; i < count; i++) {
        if(strcmp(columns[i], "ckzone") == 0) {
            is(strcmp(data[i], "ckzone"), 0, "Data is expected 'ckzone'");
        }
    }

    return 0;
}

static void
keychain_upgrade(bool musr, const char *dbname)
{
    OSStatus res;

    secd_test_setup_temp_keychain(dbname, NULL);

#if TARGET_OS_IOS
    if (musr)
        SecSecuritySetMusrMode(true, 502, 502);
#endif

#if TARGET_OS_IPHONE
    /*
     * Check system keychain migration
     */

    res = SecItemAdd((__bridge CFDictionaryRef)@{
        (id)kSecClass :  (id)kSecClassGenericPassword,
        (id)kSecAttrAccount :  @"system-label-me",
        (id)kSecUseSystemKeychain : (id)kCFBooleanTrue,
        (id)kSecValueData : [NSData dataWithBytes:"some data" length:9],
    }, NULL);
    is(res, 0, "SecItemAdd(system)");
#endif

    /*
     * Check user keychain
     */

    res = SecItemAdd((__bridge CFDictionaryRef)@{
        (id)kSecClass :  (id)kSecClassGenericPassword,
        (id)kSecAttrAccount :  @"user-label-me",
        (id)kSecValueData : [NSData dataWithBytes:"some data" length:9],
    }, NULL);
    is(res, 0, "SecItemAdd(user)");

    NSString *keychain_path = CFBridgingRelease(__SecKeychainCopyPath());

    // Add a row to a non-item table
    /* Create a new keychain sqlite db */
    sqlite3 *db = NULL;

    is(sqlite3_open([keychain_path UTF8String], &db), SQLITE_OK, "open db");
    is(sqlite3_exec(db, "INSERT into ckmirror VALUES(\"ckzone\", \"importantuuid\", \"keyuuid\", 0, \"asdf\", \"qwer\", \"ckrecord\", 0, 0, 0, NULL, NULL, NULL, \"contextID\");", NULL, NULL, NULL), SQLITE_OK, "row added to ckmirror table");
    is(sqlite3_close(db), SQLITE_OK, "close db");

    SecKeychainDbForceClose();
    SecKeychainDbReset(^{

        /* Create a new keychain sqlite db */
        sqlite3 *db;

        is(sqlite3_open([keychain_path UTF8String], &db), SQLITE_OK, "create keychain");
        is(sqlite3_exec(db, "UPDATE tversion SET minor = minor - 1", NULL, NULL, NULL), SQLITE_OK,
           "\"downgrade\" keychain");
        is(sqlite3_close(db), SQLITE_OK, "close db");
    });

#if TARGET_OS_IPHONE
    res = SecItemCopyMatching((__bridge CFDictionaryRef)@{
        (id)kSecClass :  (id)kSecClassGenericPassword,
        (id)kSecAttrAccount :  @"system-label-me",
        (id)kSecUseSystemKeychain : (id)kCFBooleanTrue,
    }, NULL);
    is(res, 0, "SecItemCopyMatching(system)");
#endif

    res = SecItemCopyMatching((__bridge CFDictionaryRef)@{
        (id)kSecClass :  (id)kSecClassGenericPassword,
        (id)kSecAttrAccount :  @"user-label-me",
    }, NULL);
    is(res, 0, "SecItemCopyMatching(user)");

    char* err = NULL;

    is(sqlite3_open([keychain_path UTF8String], &db), SQLITE_OK, "open db");
    is(sqlite3_exec(db, "select * from ckmirror;", ckmirror_row_callback, NULL, &err), SQLITE_OK, "row added to ckmirror table");
    is(sqlite3_close(db), SQLITE_OK, "close db");
    is(ckmirror_row_exists, 1, "SQLite found a row in the ckmirror table");

#if TARGET_OS_IOS
    if (musr)
        SecSecuritySetMusrMode(false, 501, -1);
#endif

    secd_test_teardown_delete_temp_keychain(dbname);
}

int
secd_20_keychain_upgrade(int argc, char *const *argv)
{
#if TARGET_OS_IPHONE
#define have_system_keychain_tests 2
#else
#define have_system_keychain_tests 0
#endif

    plan_tests((kSecdTestSetupTestCount + 5 + have_system_keychain_tests + 8) * 2);

    CFArrayRef currentACL = CFRetainSafe(SecAccessGroupsGetCurrent());

    NSMutableArray *newACL = [NSMutableArray arrayWithArray:(__bridge NSArray *)currentACL];
    [newACL addObjectsFromArray:@[
         @"com.apple.private.system-keychain",
         @"com.apple.private.syncbubble-keychain",
         @"com.apple.private.migrate-musr-system-keychain",
    ]];

    SecAccessGroupsSetCurrent((__bridge CFArrayRef)newACL);

    keychain_upgrade(false, "secd_20_keychain_upgrade");
    keychain_upgrade(true,  "secd_20_keychain_upgrade-musr");

    SecAccessGroupsSetCurrent(currentACL);
    CFReleaseNull(currentACL);

    return 0;
}
