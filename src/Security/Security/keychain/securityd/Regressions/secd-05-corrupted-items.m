/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 3, 2024.
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
#import "secd_regressions.h"

#import <Foundation/Foundation.h>
#import "keychain/securityd/SecDbItem.h"
#import <utilities/array_size.h>
#import <utilities/SecCFWrappers.h>
#import <utilities/SecFileLocations.h>
#import <utilities/fileIo.h>

#import "keychain/securityd/SecItemServer.h"

#import <Security/SecBasePriv.h>

#import <AssertMacros.h>

#import <stdio.h>
#import <unistd.h>
#import <sys/stat.h>
#import <pthread.h>

#import "SecdTestKeychainUtilities.h"

#define N_ITEMS (100)
#define N_THREADS (10)
#define N_ADDS (20)

static void *do_add(void *arg)
{
    int tid=(int)(arg);

    for(int i=0;i<N_ADDS;i++) {
        /* Creating a password */
        SInt32 v_eighty = (tid+1)*1000+i;
        CFNumberRef eighty = CFNumberCreate(NULL, kCFNumberSInt32Type, &v_eighty);
        const char *v_data = "test";
        CFDataRef pwdata = CFDataCreate(NULL, (UInt8 *)v_data, strlen(v_data));
        const void *keys[] = {
            kSecClass,
            kSecAttrServer,
            kSecAttrAccount,
            kSecAttrPort,
            kSecAttrProtocol,
            kSecAttrAuthenticationType,
            kSecValueData
        };
        const void *values[] = {
            kSecClassInternetPassword,
            CFSTR("members.spamcop.net"),
            CFSTR("smith"),
            eighty,
            CFSTR("http"),
            CFSTR("dflt"),
            pwdata
        };

        CFDictionaryRef item = CFDictionaryCreate(NULL, keys, values,
                                                  array_size(keys), NULL, NULL);

        ok_status(SecItemAdd(item, NULL), "add internet password");
        CFReleaseNull(eighty);
        CFReleaseNull(pwdata);
        CFReleaseNull(item);
    }

    return NULL;
}


int secd_05_corrupted_items(int argc, char *const *argv)
{
    plan_tests(2 + N_THREADS*(N_ADDS+1) + N_ITEMS*4 + kSecdTestSetupTestCount);
    
    /* custom keychain dir */
    secd_test_setup_temp_keychain("secd_05_corrupted_items", NULL);

    /* add a password */
    const char *v_data = "test";
    CFDataRef pwdata = CFDataCreate(NULL, (UInt8 *)v_data, strlen(v_data));
    CFMutableDictionaryRef query = CFDictionaryCreateMutable(NULL, 0, NULL, NULL);
    CFDictionaryAddValue(query, kSecClass, kSecClassInternetPassword);
    CFDictionaryAddValue(query, kSecAttrServer, CFSTR("corrupt.spamcop.net"));
    CFDictionaryAddValue(query, kSecAttrAccount, CFSTR("smith"));
    CFDictionaryAddValue(query, kSecAttrProtocol, kSecAttrProtocolHTTP);
    CFDictionaryAddValue(query, kSecAttrAuthenticationType, kSecAttrAuthenticationTypeDefault);
    CFDictionaryAddValue(query, kSecValueData, pwdata);

    SInt32 i;
    for(i=1; i<=N_ITEMS; i++) {
        CFNumberRef port = CFNumberCreate(NULL, kCFNumberSInt32Type, &i);
        CFDictionarySetValue(query, kSecAttrPort, port);
        ok_status(SecItemAdd(query, NULL), "add internet password");
        CFReleaseNull(port);
    }

    SecKeychainDbForceClose();
    SecKeychainDbReset(^{
        /* corrupt all the password */
        NSString *keychain_path = CFBridgingRelease(__SecKeychainCopyPath());
        char corrupt_item_sql[80];
        sqlite3 *db;

        is(sqlite3_open([keychain_path UTF8String], &db), SQLITE_OK, "open keychain");

        for(int i=1;i<=N_ITEMS;i++) {
            ok_unix(snprintf(corrupt_item_sql, sizeof(corrupt_item_sql), "UPDATE inet SET data=X'12345678' WHERE rowid=%d", i));
            is(sqlite3_exec(db, corrupt_item_sql, NULL, NULL, NULL), SQLITE_OK, "corrupting keychain item");
        }

        is(sqlite3_close_v2(db), SQLITE_OK,
           "Should be able to close db");
    });

    /* start the adder threads */
    pthread_t add_thread[N_THREADS];
    void *add_err[N_THREADS] = {NULL,};

    for(int i=0; i<N_THREADS; i++)
        pthread_create(&add_thread[i], NULL, do_add, (void*)(intptr_t)i);

    /* query the corrupted items */
    CFDictionaryAddValue(query, kSecReturnPersistentRef, kCFBooleanTrue);
    for(int i=1;i<=N_ITEMS;i++) {
        CFTypeRef ref = NULL;
        CFNumberRef port = CFNumberCreate(NULL, kCFNumberSInt32Type, &i);
        CFDictionarySetValue(query, kSecAttrPort, port);
        is_status(SecItemCopyMatching(query, &ref), errSecItemNotFound, "Item not found");
        CFReleaseNull(port);
        CFReleaseNull(ref);
    }

    /* collect the adder threads */
    for(int i=0; i<N_THREADS; i++)
        pthread_join(add_thread[i], &add_err[i]);

    for(int i=0; i<N_THREADS; i++)
        ok(add_err[i]==NULL, "add thread");

    CFReleaseNull(pwdata);
    CFReleaseNull(query);

    secd_test_teardown_delete_temp_keychain("secd_05_corrupted_items");
    return 0;
}
