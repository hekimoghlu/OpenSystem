/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 19, 2022.
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
#import "SecKnownFilePaths.h"
#import "OSX/utilities/SecCFRelease.h"

// This file is separate from SecFileLocation.c because it has a global variable.
// We need exactly one of those per address space, so it needs to live in the Security framework.
static CFURLRef sCustomHomeURL = NULL;

CFURLRef SecCopyHomeURL(void)
{
    // This returns a CFURLRef so that it can be passed as the second parameter
    // to CFURLCreateCopyAppendingPathComponent

    CFURLRef homeURL = sCustomHomeURL;
    if (homeURL) {
        CFRetain(homeURL);
    } else {
        homeURL = CFCopyHomeDirectoryURL();
    }

    return homeURL;
}

CFURLRef SecCopyBaseFilesURL(bool system)
{
    CFURLRef baseURL = sCustomHomeURL;
    if (baseURL) {
        CFRetain(baseURL);
    } else {
#if TARGET_OS_OSX
        if (system) {
            baseURL = CFURLCreateWithFileSystemPath(NULL, CFSTR("/"), kCFURLPOSIXPathStyle, true);
        } else {
            baseURL = SecCopyHomeURL();
        }
#elif TARGET_OS_SIMULATOR
        baseURL = SecCopyHomeURL();
#else
        if (system) {
            baseURL = CFURLCreateWithFileSystemPath(NULL, CFSTR("/"), kCFURLPOSIXPathStyle, true);
        } else {
            baseURL = SecCopyHomeURL();
        }
#endif
    }
    return baseURL;
}

void SecSetCustomHomeURL(CFURLRef url)
{
    sCustomHomeURL = CFRetainSafe(url);
}

void SecSetCustomHomeURLString(CFStringRef home_path)
{
    CFReleaseNull(sCustomHomeURL);
    if (home_path) {
        sCustomHomeURL = CFURLCreateWithFileSystemPath(NULL, home_path, kCFURLPOSIXPathStyle, true);
    }
}
