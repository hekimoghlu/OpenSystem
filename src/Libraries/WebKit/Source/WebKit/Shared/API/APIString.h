/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 12, 2023.
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

#include "APIObject.h"
#include <wtf/text/StringView.h>

namespace API {

class String final : public ObjectImpl<Object::Type::String> {
public:
    static Ref<String> createNull()
    {
        return adoptRef(*new String);
    }

    static Ref<String> create(WTF::String&& string)
    {
        return adoptRef(*new String(string.isNull() ? WTF::String(StringImpl::empty()) : WTFMove(string).isolatedCopy()));
    }

    static Ref<String> create(const WTF::String& string)
    {
        return create(string.isolatedCopy());
    }

    virtual ~String()
    {
    }

    WTF::StringView stringView() const { return m_string; }
    WTF::String string() const { return m_string.isolatedCopy(); }

private:
    String()
        : m_string()
    {
    }

    String(WTF::String&& string)
        : m_string(WTFMove(string))
    {
        ASSERT(!m_string.isNull());
        ASSERT(m_string.isSafeToSendToAnotherThread());
    }

    const WTF::String m_string;
};

} // namespace WebKit

SPECIALIZE_TYPE_TRAITS_API_OBJECT(String);
