/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 10, 2024.
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
#include <WebCore/CustomHeaderFields.h>

namespace API {

class CustomHeaderFields final : public ObjectImpl<Object::Type::CustomHeaderFields> {
public:
    template<typename... Args> static Ref<CustomHeaderFields> create(Args&&... args)
    {
        return adoptRef(*new CustomHeaderFields(std::forward<Args>(args)...));
    }

    CustomHeaderFields() = default;

    const Vector<WebCore::HTTPHeaderField>& fields() const { return m_fields.fields; }
    void setFields(Vector<WebCore::HTTPHeaderField>&& fields) { m_fields.fields = WTFMove(fields); }

    const Vector<WTF::String> thirdPartyDomains() const { return m_fields.thirdPartyDomains; }
    void setThirdPartyDomains(Vector<WTF::String>&& domains) { m_fields.thirdPartyDomains = WTFMove(domains); }

    const WebCore::CustomHeaderFields& coreFields() const { return m_fields; }

private:
    CustomHeaderFields(const WebCore::CustomHeaderFields& fields)
        : m_fields(fields) { }

    WebCore::CustomHeaderFields m_fields;
};

} // namespace API

SPECIALIZE_TYPE_TRAITS_API_OBJECT(CustomHeaderFields);
