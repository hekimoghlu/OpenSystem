/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 7, 2021.
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
#pragma once

#include <wtf/StdLibExtras.h>
#include <wtf/cf/VectorCF.h>
#include <wtf/text/StringConcatenate.h>

#if USE(CF)

namespace WTF {

template<> class StringTypeAdapter<CFStringRef> {
public:
    StringTypeAdapter(CFStringRef);
    unsigned length() const { return m_string ? CFStringGetLength(m_string) : 0; }
    bool is8Bit() const { return !m_string || CFStringGetCStringPtr(m_string, kCFStringEncodingISOLatin1); }
    template<typename CharacterType> void writeTo(std::span<CharacterType>) const;

private:
    CFStringRef m_string;
};

inline StringTypeAdapter<CFStringRef>::StringTypeAdapter(CFStringRef string)
    : m_string { string }
{
}

template<> inline void StringTypeAdapter<CFStringRef>::writeTo<LChar>(std::span<LChar> destination) const
{
    if (m_string)
        memcpySpan(destination, CFStringGetLatin1CStringSpan(m_string));
}

template<> inline void StringTypeAdapter<CFStringRef>::writeTo<UChar>(std::span<UChar> destination) const
{
    if (m_string)
        CFStringGetCharacters(m_string, CFRangeMake(0, CFStringGetLength(m_string)), reinterpret_cast<UniChar*>(destination.data()));
}

#ifdef __OBJC__

template<> class StringTypeAdapter<NSString *> : public StringTypeAdapter<CFStringRef> {
public:
    StringTypeAdapter(NSString *string)
        : StringTypeAdapter<CFStringRef>((__bridge CFStringRef)string)
    {
    }
};

#endif

}

#endif // USE(CF)
