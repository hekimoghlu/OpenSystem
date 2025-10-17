/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 7, 2022.
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
#include <WebCore/SecurityOrigin.h>
#include <wtf/Ref.h>

namespace API {

class SecurityOrigin : public API::ObjectImpl<API::Object::Type::SecurityOrigin> {
public:
    static Ref<SecurityOrigin> createFromString(const WTF::String& string)
    {
        return adoptRef(*new SecurityOrigin(WebCore::SecurityOriginData::fromURLWithoutStrictOpaqueness(WTF::URL { string })));
    }

    static Ref<SecurityOrigin> create(const WTF::String& protocol, const WTF::String& host, std::optional<uint16_t> port)
    {
        return adoptRef(*new SecurityOrigin({ protocol, host, port }));
    }

    static Ref<SecurityOrigin> create(const WebCore::SecurityOrigin& securityOrigin)
    {
        return adoptRef(*new SecurityOrigin(securityOrigin.data().isolatedCopy()));
    }

    static Ref<SecurityOrigin> create(const WebCore::SecurityOriginData& securityOriginData)
    {
        return adoptRef(*new SecurityOrigin(securityOriginData.isolatedCopy()));
    }

    const WebCore::SecurityOriginData& securityOrigin() const { return m_securityOrigin; }

private:
    SecurityOrigin(WebCore::SecurityOriginData&& securityOrigin)
        : m_securityOrigin(WTFMove(securityOrigin))
    {
    }

    WebCore::SecurityOriginData m_securityOrigin;
};

}
