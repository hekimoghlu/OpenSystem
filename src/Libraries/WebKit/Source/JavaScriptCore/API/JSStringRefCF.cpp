/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 19, 2025.
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
#include "config.h"
#include "JSStringRefCF.h"

#include "APICast.h"
#include "InitializeThreading.h"
#include "JSCJSValue.h"
#include "JSStringRef.h"
#include "OpaqueJSString.h"
#include <wtf/StdLibExtras.h>

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

JSStringRef JSStringCreateWithCFString(CFStringRef string)
{
    JSC::initialize();

    // We cannot use CFIndex here since CFStringGetLength can return values larger than
    // it can hold.  (<rdar://problem/6806478>)
    size_t length = CFStringGetLength(string);
    if (!length)
        return &OpaqueJSString::create(""_span8).leakRef();

    Vector<LChar, 1024> lcharBuffer(length);
    CFIndex usedBufferLength;
    CFIndex convertedSize = CFStringGetBytes(string, CFRangeMake(0, length), kCFStringEncodingISOLatin1, 0, false, lcharBuffer.data(), length, &usedBufferLength);
    if (static_cast<size_t>(convertedSize) == length && static_cast<size_t>(usedBufferLength) == length)
        return &OpaqueJSString::create(lcharBuffer.span()).leakRef();

    Vector<UniChar> buffer(length);
    CFStringGetCharacters(string, CFRangeMake(0, length), buffer.data());
    static_assert(sizeof(UniChar) == sizeof(UChar), "UniChar and UChar must be same size");
    return &OpaqueJSString::create({ reinterpret_cast<UChar*>(buffer.data()), length }).leakRef();
}

CFStringRef JSStringCopyCFString(CFAllocatorRef allocator, JSStringRef string)
{
    if (!string || !string->length())
        return CFSTR("");

    if (string->is8Bit()) {
        auto characters = string->span8();
        return CFStringCreateWithBytes(allocator, characters.data(), characters.size(), kCFStringEncodingISOLatin1, false);
    }
    auto characters = string->span16();
    return CFStringCreateWithCharacters(allocator, reinterpret_cast<const UniChar*>(characters.data()), characters.size());
}

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
