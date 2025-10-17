/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 9, 2024.
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

#include "builtin_commands.h"


int
command_whoami(__unused int argc, __unused char * const * argv)
{
    @autoreleasepool {
        CFErrorRef error = NULL;
        NSDictionary *dict = NULL;

        dict = CFBridgingRelease(_SecSecuritydCopyWhoAmI(&error));
        if (dict) {
            puts([[NSString stringWithFormat:@"the server thinks we are:\n%@\n", dict] UTF8String]);
        } else {
            puts([[NSString stringWithFormat:@"no reply from server: %@", error] UTF8String]);
        }
        if (error)
            CFRelease(error);
    }

    return 0;
}
