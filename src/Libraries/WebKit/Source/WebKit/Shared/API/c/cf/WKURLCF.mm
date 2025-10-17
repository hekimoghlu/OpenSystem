/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 9, 2023.
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
#import "config.h"
#import "WKURLCF.h"

#import "WKAPICast.h"
#import "WKNSURL.h"
#import <objc/runtime.h>
#import <wtf/cf/CFURLExtras.h>

static inline Class wkNSURLClass()
{
    static dispatch_once_t once;
    static Class wkNSURLClass;
    dispatch_once(&once, ^{
        wkNSURLClass = [WKNSURL class];
    });
    return wkNSURLClass;
}

WKURLRef WKURLCreateWithCFURL(CFURLRef cfURL)
{
    if (!cfURL)
        return nullptr;

    // Since WKNSURL is an internal class with no subclasses, we can do a simple equality check.
    if (object_getClass((__bridge NSURL *)cfURL) == wkNSURLClass())
        return WebKit::toAPI(downcast<API::URL>(&[(WKNSURL *)(__bridge NSURL *)CFRetain(cfURL) _apiObject]));

    // FIXME: Why is it OK to ignore the base URL in the CFURL here?
    return WebKit::toCopiedURLAPI(bytesAsString(cfURL));
}

CFURLRef WKURLCopyCFURL(CFAllocatorRef allocatorRef, WKURLRef URLRef)
{
    auto& string = WebKit::toImpl(URLRef)->string();
    if (string.isNull())
        return nullptr;

    // We first create a CString and then create the CFURL from it. This will ensure that the CFURL is stored in 
    // UTF-8 which uses less memory and is what WebKit clients might expect.

    auto buffer = string.utf8();
    auto bufferSpan = buffer.span();
    return CFURLCreateAbsoluteURLWithBytes(nullptr, bufferSpan.data(), bufferSpan.size(), kCFStringEncodingUTF8, nullptr, true);
}
