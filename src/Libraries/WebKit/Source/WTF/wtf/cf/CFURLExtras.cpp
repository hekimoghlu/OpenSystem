/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 20, 2022.
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
#include <wtf/cf/CFURLExtras.h>

#include <wtf/URL.h>

namespace WTF {

RetainPtr<CFDataRef> bytesAsCFData(std::span<const uint8_t> bytes)
{
    return adoptCF(CFDataCreate(nullptr, bytes.data(), bytes.size()));
}

RetainPtr<CFDataRef> bytesAsCFData(CFURLRef url)
{
    if (!url)
        return nullptr;
    auto bytesLength = CFURLGetBytes(url, nullptr, 0);
    RELEASE_ASSERT(bytesLength != -1);
    auto buffer = static_cast<uint8_t*>(malloc(bytesLength));
    RELEASE_ASSERT(buffer);
    CFURLGetBytes(url, buffer, bytesLength);
    return adoptCF(CFDataCreateWithBytesNoCopy(nullptr, buffer, bytesLength, kCFAllocatorMalloc));
}

String bytesAsString(CFURLRef url)
{
    if (!url)
        return { };
    auto bytesLength = CFURLGetBytes(url, nullptr, 0);
    RELEASE_ASSERT(bytesLength != -1);
    RELEASE_ASSERT(bytesLength <= static_cast<CFIndex>(String::MaxLength));
    std::span<LChar> buffer;
    auto result = String::createUninitialized(bytesLength, buffer);
    CFURLGetBytes(url, buffer.data(), buffer.size());
    return result;
}

Vector<uint8_t, URLBytesVectorInlineCapacity> bytesAsVector(CFURLRef url)
{
    if (!url)
        return { };

    Vector<uint8_t, URLBytesVectorInlineCapacity> result(URLBytesVectorInlineCapacity);
    auto bytesLength = CFURLGetBytes(url, result.data(), URLBytesVectorInlineCapacity);
    if (bytesLength != -1)
        result.shrink(bytesLength);
    else {
        bytesLength = CFURLGetBytes(url, nullptr, 0);
        RELEASE_ASSERT(bytesLength != -1);
        result.grow(bytesLength);
        CFURLGetBytes(url, result.data(), bytesLength);
    }

    // This may look like it copies the bytes in the vector, but due to the return value optimization it does not.
    return result;
}

bool isSameOrigin(CFURLRef a, const URL& b)
{
    ASSERT(b.protocolIsInHTTPFamily());

    if (b.hasCredentials())
        return protocolHostAndPortAreEqual(a, b);

    auto aBytes = bytesAsVector(a);
    RELEASE_ASSERT(aBytes.size() <= String::MaxLength);

    StringView aString { aBytes.span() };
    StringView bString { b.string() };

    if (!b.hasPath())
        return aString == bString;

    unsigned afterPathSeparator = b.pathStart() + 1;
    return aString.left(afterPathSeparator) == bString.left(afterPathSeparator);
}

}
