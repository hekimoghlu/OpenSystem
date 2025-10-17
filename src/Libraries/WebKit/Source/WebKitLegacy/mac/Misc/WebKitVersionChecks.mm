/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 11, 2023.
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
#import "WebKitVersionChecks.h"

#import <mach-o/dyld.h>
#import <mutex>

static int WebKitLinkTimeVersion(void);
static int overriddenWebKitLinkTimeVersion;

BOOL WebKitLinkedOnOrAfter(int version)
{
#if !PLATFORM(IOS_FAMILY)
    return (WebKitLinkTimeVersion() >= version);
#else
    int32_t linkTimeVersion = WebKitLinkTimeVersion();
    int32_t majorVersion = linkTimeVersion >> 16 & 0x0000FFFF;
    
    // The application was not linked against UIKit so assume most recent WebKit
    if (linkTimeVersion == -1)
        return YES;
    
    return (majorVersion >= version);
#endif
}

void setWebKitLinkTimeVersion(int version)
{
    overriddenWebKitLinkTimeVersion = version;
}

static int WebKitLinkTimeVersion(void)
{
    if (overriddenWebKitLinkTimeVersion)
        return overriddenWebKitLinkTimeVersion;

#if !PLATFORM(IOS_FAMILY)
    return NSVersionOfLinkTimeLibrary("WebKit");
#else
    // <rdar://problem/6627758> Need to implement WebKitLinkedOnOrAfter
    // Third party applications do not link against WebKit, but rather against UIKit.
    return NSVersionOfLinkTimeLibrary("UIKit");
#endif
}
