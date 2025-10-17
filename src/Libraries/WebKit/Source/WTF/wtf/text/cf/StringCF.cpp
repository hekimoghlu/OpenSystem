/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 12, 2022.
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
#include <wtf/text/WTFString.h>

#if USE(CF)

#include <CoreFoundation/CoreFoundation.h>
#include <wtf/RetainPtr.h>
#include <wtf/text/StringBuffer.h>

namespace WTF {

String::String(CFStringRef str)
{
    if (!str)
        return;

    CFIndex size = CFStringGetLength(str);
    if (!size) {
        m_impl = StringImpl::empty();
        return;
    }

    {
        StringBuffer<LChar> buffer(size);
        CFIndex usedBufLen;
        CFIndex convertedSize = CFStringGetBytes(str, CFRangeMake(0, size), kCFStringEncodingISOLatin1, 0, false, buffer.characters(), size, &usedBufLen);
        if (convertedSize == size && usedBufLen == size) {
            m_impl = StringImpl::adopt(WTFMove(buffer));
            return;
        }
    }

    StringBuffer<UChar> ucharBuffer(size);
    CFStringGetCharacters(str, CFRangeMake(0, size), reinterpret_cast<UniChar *>(ucharBuffer.characters()));
    m_impl = StringImpl::adopt(WTFMove(ucharBuffer));
}

RetainPtr<CFStringRef> String::createCFString() const
{
    if (!m_impl)
        return CFSTR("");

    return m_impl->createCFString();
}

RetainPtr<CFStringRef> makeCFArrayElement(const String& vectorElement)
{
    return vectorElement.createCFString();
}

std::optional<String> makeVectorElement(const String*, CFStringRef cfString)
{
    if (cfString)
        return { { cfString } };
    return std::nullopt;
}

}

#endif // USE(CF)
