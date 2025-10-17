/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 16, 2024.
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
#import <Security/Security.h>
#import <Security/SecItemPriv.h>
#import <err.h>

#include "builtin_commands.h"


int
command_bubble(__unused int argc, __unused char * const * argv)
{
    @autoreleasepool {
        CFErrorRef error = NULL;
        uid_t uid;

        if (argc < 2)
            errx(1, "missing uid argument");

        uid = atoi(argv[1]);
        if (uid == 0)
            errx(1, "syncbubble for root not supported");

        NSArray *services = @[@"com.apple.cloudd.sync", @"com.apple.mailq.sync"];

        if (_SecSyncBubbleTransfer((__bridge CFArrayRef)services, uid, &error)) {
            errx(0, "%s", [[NSString stringWithFormat:@"sync bubble populated"] UTF8String]);
        } else {
            errx(1, "%s", [[NSString stringWithFormat:@"sync bubble failed to inflate: %@", error] UTF8String]);
        }
    }
}

int
command_system_transfer(__unused int argc, __unused char * const * argv)
{
    @autoreleasepool {
        CFErrorRef error = NULL;
        if (_SecSystemKeychainTransfer(&error)) {
            errx(0, "transferred to system keychain");
        } else {
            errx(1, "%s", [[NSString stringWithFormat:@"failed to transfer to system keychain: %@", error] UTF8String]);
        }
    }
}

int
command_system_transcrypt(__unused int argc, __unused char * const * argv)
{
    @autoreleasepool {
        CFErrorRef error = NULL;
        if (_SecSystemKeychainTranscrypt(&error)) {
            errx(0, "transcrypted to system keychain");
        } else {
            errx(1, "%s", [[NSString stringWithFormat:@"failed to transcrypt to system keychain: %@", error] UTF8String]);
        }
    }
}
