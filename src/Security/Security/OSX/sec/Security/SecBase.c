/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 9, 2021.
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
/*
 * SecBase.c
 */

#ifdef STANDALONE
/* Allows us to build genanchors against the BaseSDK. */
#undef __ENVIRONMENT_IPHONE_OS_VERSION_MIN_REQUIRED__
#undef __ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__
#endif

#include <Availability.h>
#include "SecFramework.h"
#include <dispatch/dispatch.h>
#include <CoreFoundation/CFBundle.h>
#include <CoreFoundation/CFURLAccess.h>
#include <Security/SecRandom.h>
#include <CommonCrypto/CommonRandomSPI.h>
#include <fcntl.h>
#include <sys/types.h>
#include <unistd.h>
#include <utilities/debugging.h>
#include <utilities/SecCFWrappers.h>
#include <Security/SecBase.h>
#include <inttypes.h>
#include <Security/SecFramework.h>

#if !TARGET_OS_OSX

static CFStringRef copyErrorMessageFromBundle(OSStatus status, CFStringRef tableName);

// caller MUST release the string, since it is gotten with "CFCopyLocalizedStringFromTableInBundle"
// intended use of reserved param is to pass in CFStringRef with name of the Table for lookup
// Will look by default in "SecErrorMessages.strings" in the resources of Security.framework.

CFStringRef
SecCopyErrorMessageString(OSStatus status, void *reserved)
{
    CFStringRef result = copyErrorMessageFromBundle(status, CFSTR("SecErrorMessages"));
    if (!result)
        result = copyErrorMessageFromBundle(status, CFSTR("SecDebugErrorMessages"));

    if (!result)
    {
        // no error message found, so format a faked-up error message from the status
        result = CFStringCreateWithFormat(NULL, NULL, CFSTR("OSStatus %d"), (int)status);
    }

    return result;
}

CFStringRef
copyErrorMessageFromBundle(OSStatus status,CFStringRef tableName)
{

    CFStringRef errorString = nil;
    CFStringRef keyString = nil;
    CFBundleRef secBundle = NULL;

    // Make a bundle instance using the URLRef.
    secBundle = SecFrameworkGetBundle();
    if (!secBundle)
        goto exit;

    // Convert status to Int32 string representation, e.g. "-25924"
    keyString = CFStringCreateWithFormat (kCFAllocatorDefault, NULL, CFSTR("%d"), (int)status);
    if (!keyString)
        goto exit;

    errorString = CFCopyLocalizedStringFromTableInBundle(keyString, tableName, secBundle, NULL);
    if (CFStringCompare(errorString, keyString, 0) == kCFCompareEqualTo)    // no real error message
    {
        if (errorString)
            CFRelease(errorString);
        errorString = nil;
    }
exit:
    if (keyString)
        CFRelease(keyString);

    return errorString;
}

const SecRandomRef kSecRandomDefault = NULL;

int SecRandomCopyBytes(__unused SecRandomRef rnd, size_t count, void *bytes) {
    return CCRandomCopyBytes(kCCRandomDefault, bytes, count);
}

#endif // TARGET_OS_OSX
