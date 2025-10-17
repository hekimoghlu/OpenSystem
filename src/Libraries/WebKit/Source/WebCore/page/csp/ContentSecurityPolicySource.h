/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 31, 2024.
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

#include <wtf/TZoneMalloc.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class ContentSecurityPolicy;
class SecurityOriginData;

enum class IsSelfSource : bool { No, Yes };

class ContentSecurityPolicySource {
    WTF_MAKE_TZONE_ALLOCATED(ContentSecurityPolicySource);
public:
    ContentSecurityPolicySource(const ContentSecurityPolicy&, const String& scheme, const String& host, std::optional<uint16_t> port, const String& path, bool hostHasWildcard, bool portHasWildcard, IsSelfSource);

    bool matches(const URL&, bool didReceiveRedirectResponse = false) const;

    operator SecurityOriginData() const;

private:
    bool schemeMatches(const URL&) const;
    bool hostMatches(const URL&) const;
    bool pathMatches(const URL&) const;
    bool portMatches(const URL&) const;
    bool isSchemeOnly() const;

    const ContentSecurityPolicy& m_policy;
    String m_scheme;
    String m_host;
    String m_path;
    std::optional<uint16_t> m_port;

    bool m_hostHasWildcard;
    bool m_portHasWildcard;
    bool m_isSelfSource;
};

} // namespace WebCore
