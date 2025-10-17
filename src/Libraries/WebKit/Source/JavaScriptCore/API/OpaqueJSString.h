/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 15, 2022.
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

#include <atomic>
#include <wtf/ThreadSafeRefCounted.h>
#include <wtf/text/WTFString.h>

namespace JSC {
    class Identifier;
    class VM;
}

struct OpaqueJSString : public ThreadSafeRefCounted<OpaqueJSString> {
    static Ref<OpaqueJSString> create()
    {
        return adoptRef(*new OpaqueJSString);
    }

    static Ref<OpaqueJSString> create(std::span<const LChar> characters)
    {
        return adoptRef(*new OpaqueJSString(characters));
    }

    static Ref<OpaqueJSString> create(std::span<const UChar> characters)
    {
        return adoptRef(*new OpaqueJSString(characters));
    }

    JS_EXPORT_PRIVATE static RefPtr<OpaqueJSString> tryCreate(const String&);
    JS_EXPORT_PRIVATE static RefPtr<OpaqueJSString> tryCreate(String&&);

    JS_EXPORT_PRIVATE ~OpaqueJSString();

    bool is8Bit() { return m_string.is8Bit(); }
    std::span<const LChar> span8() { return m_string.span8(); }
    std::span<const UChar> span16() { return m_string.span16(); }
    unsigned length() { return m_string.length(); }

    const UChar* characters();

    JS_EXPORT_PRIVATE String string() const;
    JSC::Identifier identifier(JSC::VM*) const;

    static bool equal(const OpaqueJSString*, const OpaqueJSString*);

private:
    friend class WTF::ThreadSafeRefCounted<OpaqueJSString>;

    OpaqueJSString()
        : m_characters(nullptr)
    {
    }

    OpaqueJSString(const String& string)
        : m_string(string.isolatedCopy())
        , m_characters(m_string.impl() && m_string.is8Bit() ? nullptr : const_cast<UChar*>(m_string.span16().data()))
    {
    }

    explicit OpaqueJSString(String&& string)
        : m_string(WTFMove(string))
        , m_characters(m_string.impl() && m_string.is8Bit() ? nullptr : const_cast<UChar*>(m_string.span16().data()))
    {
    }

    OpaqueJSString(std::span<const LChar> characters)
        : m_string(characters)
        , m_characters(nullptr)
    {
    }

    OpaqueJSString(std::span<const UChar> characters)
        : m_string(characters)
        , m_characters(m_string.impl() && m_string.is8Bit() ? nullptr : const_cast<UChar*>(m_string.span16().data()))
    {
    }

    String m_string;

    // This will be initialized on demand when characters() is called if the string needs up-conversion.
    std::atomic<UChar*> m_characters;
};
