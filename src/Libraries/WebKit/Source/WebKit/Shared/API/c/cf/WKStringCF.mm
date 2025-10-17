/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 23, 2023.
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
#import "WKStringCF.h"

#import "WKAPICast.h"
#import "WKNSString.h"
#import <objc/runtime.h>
#import <wtf/text/WTFString.h>

static inline Class wkNSStringClass()
{
    static dispatch_once_t once;
    static Class wkNSStringClass;
    dispatch_once(&once, ^{
        wkNSStringClass = [WKNSString class];
    });
    return wkNSStringClass;
}

WKStringRef WKStringCreateWithCFString(CFStringRef cfString)
{
    // Since WKNSString is an internal class with no subclasses, we can do a simple equality check.
    if (object_getClass((__bridge NSString *)cfString) == wkNSStringClass())
        return WebKit::toAPI(downcast<API::String>(&[(WKNSString *)(__bridge NSString *)CFRetain(cfString) _apiObject]));
    String string(cfString);
    return WebKit::toCopiedAPI(string);
}

CFStringRef WKStringCopyCFString(CFAllocatorRef allocatorRef, WKStringRef stringRef)
{
    ASSERT(!WebKit::toImpl(stringRef)->string().isNull());

    auto string = WebKit::toImpl(stringRef)->string();

    // NOTE: This does not use StringImpl::createCFString() since that function
    // expects to be called on the thread running WebCore.
    if (string.is8Bit()) {
        auto characters = string.span8();
        return CFStringCreateWithBytes(allocatorRef, characters.data(), characters.size(), kCFStringEncodingISOLatin1, true);
    }
    auto characters = string.span16();
    return CFStringCreateWithCharacters(allocatorRef, reinterpret_cast<const UniChar*>(characters.data()), characters.size());
}
