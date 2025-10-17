/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 23, 2025.
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
 * SecFramework.c - generic non API class specific functions
 */

#ifdef STANDALONE
/* Allows us to build genanchors against the BaseSDK. */
#undef __ENVIRONMENT_IPHONE_OS_VERSION_MIN_REQUIRED__
#undef __ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__
#endif

#include "SecFramework.h"
#include <CoreFoundation/CFBundle.h>
#include "utilities/SecCFWrappers.h"
#include <dispatch/dispatch.h>
#include <stdlib.h>
#include <string.h>

/* Security.framework's bundle id. */
#if TARGET_OS_IPHONE
CFStringRef kSecFrameworkBundleID = CFSTR("com.apple.Security");
#else
CFStringRef kSecFrameworkBundleID = CFSTR("com.apple.security");
#endif

static CFStringRef kSecurityFrameworkBundlePath = CFSTR("/System/Library/Frameworks/Security.framework");

CFGiblisGetSingleton(CFBundleRef, SecFrameworkGetBundle, bundle,  ^{
    CFStringRef bundlePath = NULL;
#if TARGET_OS_SIMULATOR
    char *simulatorRoot = getenv("SIMULATOR_ROOT");
    if (simulatorRoot) {
        bundlePath = CFStringCreateWithFormat(NULL, NULL, CFSTR("%s%@"), simulatorRoot, kSecurityFrameworkBundlePath);
    }
#endif
    if (!bundlePath) {
        bundlePath = CFRetainSafe(kSecurityFrameworkBundlePath);
    }
    CFURLRef url = CFURLCreateWithFileSystemPath(kCFAllocatorDefault, bundlePath, kCFURLPOSIXPathStyle, true);
    *bundle = (url) ? CFBundleCreate(kCFAllocatorDefault, url) : NULL;
    CFReleaseSafe(url);
    CFReleaseSafe(bundlePath);
})

CFStringRef SecFrameworkCopyLocalizedString(CFStringRef key,
    CFStringRef tableName) {
    CFBundleRef bundle = SecFrameworkGetBundle();
    if (bundle)
        return CFBundleCopyLocalizedString(bundle, key, key, tableName);

    return CFRetainSafe(key);
}

Boolean SecFrameworkIsRunningInXcode(void) {
    static Boolean runningInXcode = false;
    static dispatch_once_t envCheckOnce = 0;
    dispatch_once(&envCheckOnce, ^{
        const char* envVar = getenv("NSUnbufferedIO");
        if (envVar != NULL && strcmp(envVar, "YES") == 0) {
            runningInXcode = true;
        }
    });
    return runningInXcode;
}
