/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 23, 2024.
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
#include <Security/SecTrustPriv.h>
#include <Security/SecTrustSettingsPriv.h>
#include <utilities/SecCFWrappers.h>

#include "SecurityCommands.h"

static int returnCFError(CFErrorRef CF_CONSUMED error) {
    CFStringRef errorString = CFErrorCopyDescription(error);
    CFStringPerformWithCString(errorString, ^(const char *utf8Str) {
        fprintf(stderr, "Failed to reset trust settings: %s\n", utf8Str);
    });
    CFIndex errCode = CFErrorGetCode(error);
    CFReleaseNull(error);
    return (int)errCode;
}

int reset_trust_settings(int argc, char * const *argv) {
    int arg;
    SecTrustResetFlags flags = 0;
    /* parse args */
    while ((arg = getopt(argc, argv, "AUXOIVC")) != -1) {
        switch(arg) {
            case 'A':
                flags = kSecTrustResetAllSettings;
                break;
            case 'U':
                flags |= kSecTrustResetUserTrustSettings;
                break;
            case 'X':
                flags |= kSecTrustResetExceptions;
                break;
            case 'O':
                flags |= kSecTrustResetOCSPCache;
                break;
            case 'I':
                flags |= kSecTrustResetIssuersCache;
                break;
            case 'V':
                flags |= kSecTrustResetValidDB;
                break;
            case 'C':
                flags |= kSecTrustResetAllCaches;
                break;
            default:
                flags = 0;
                break;
        }
    }
    if (flags == 0) {
        return SHOW_USAGE_MESSAGE; // no flags were specified
    }
    CFErrorRef error = NULL;
    bool result = SecTrustResetSettings(flags, &error);
    if (!result && error) {
        return returnCFError(error);
    }
    return 0;
}
