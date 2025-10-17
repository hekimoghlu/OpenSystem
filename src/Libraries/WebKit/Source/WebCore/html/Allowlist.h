/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 5, 2022.
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

#include "SecurityOriginData.h"

namespace WebCore {

// https://w3c.github.io/webappsec-permissions-policy/#allowlists
class Allowlist {
public:
    Allowlist() = default;
    struct AllowAllOrigins { };
    Allowlist(AllowAllOrigins allow)
        : m_origins(allow)
    {
    }
    explicit Allowlist(const SecurityOriginData& origin)
        : m_origins(HashSet<SecurityOriginData> { origin })
    {
    }
    explicit Allowlist(HashSet<SecurityOriginData>&& origins)
        : m_origins(WTFMove(origins))
    {
    }

    using OriginsVariant = std::variant<HashSet<SecurityOriginData>, AllowAllOrigins>;
    explicit Allowlist(OriginsVariant&& origins)
        : m_origins(WTFMove(origins))
    {
    }
    const OriginsVariant& origins() const { return m_origins; }

    // This is simplified version of https://w3c.github.io/webappsec-permissions-policy/#matches.
    bool matches(const SecurityOriginData& origin) const
    {
        return std::visit(WTF::makeVisitor([&origin](const HashSet<SecurityOriginData>& origins) -> bool {
            return origins.contains(origin);
        }, [&] (const auto&) {
            return true;
        }), m_origins);
    }

private:
    OriginsVariant m_origins;
};

} // namespace WebCore
