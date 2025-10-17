/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 4, 2025.
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

#include <Security/SecBase.h>
#include <sys/types.h>
#include <stdio.h>
#include <stdlib.h>
#include <malloc/malloc.h>
#include <unistd.h>
#include <Security/Security.h>
#include <Security/SecKeychainPriv.h>
#include "SecKeychainFuzzer.h"

#define TEMPFILE_TEMPLATE "/tmp/keychain_parser_fuzzer.XXXXXX"

int SecKeychainFuzzer(const uint8_t *Data, size_t Size) {
    char* temppath = strdup(TEMPFILE_TEMPLATE);

    int fd = mkstemp(temppath);
    if (fd < 0) {
        fprintf(stderr, "Unable to create tempfile: %d\n", errno);
        free(temppath);
        return 0;
    }

    size_t written = write(fd, Data, Size);
    if (written != Size) {
        fprintf(stderr, "Failed to write all bytes to tempfile\n");
    } else {
        SecKeychainRef keychain_ref = NULL;
        OSStatus status = SecKeychainOpen(temppath, &keychain_ref);
        if(status == errSecSuccess && keychain_ref != NULL) {
            SecKeychainStatus kcStatus;
            SecKeychainGetStatus(keychain_ref, &kcStatus);

            UInt32 version = 0;
            SecKeychainGetKeychainVersion(keychain_ref, &version);

            Boolean is_valid = false;
            SecKeychainIsValid(keychain_ref, &is_valid);

            UInt32 passwordLength = 0;
            void *passwordData = NULL;
            SecKeychainItemRef itemRef = NULL;
            SecKeychainFindGenericPassword(keychain_ref, 10, "SurfWriter", 10, "MyUserAcct", &passwordLength, &passwordData, &itemRef);

            if(passwordData != NULL) {
                SecKeychainItemFreeContent(NULL, passwordData);
            }

            if(itemRef != NULL) {
                CFRelease(itemRef);
            }

            CFRelease(keychain_ref);
        } else {
            fprintf(stderr, "Keychain parsing error! %d %p\n", status, keychain_ref);
        }
    }

    if (remove(temppath) != 0) {
        fprintf(stderr, "Unable to remove tempfile: %s\n", temppath);
    }

    if (temppath) {
        free(temppath);
    }

    return 0;
}
