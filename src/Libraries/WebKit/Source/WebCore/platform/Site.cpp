/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 9, 2023.
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
#include "Site.h"

#include <wtf/HashFunctions.h>

namespace WebCore {

Site::Site(const URL& url)
    : m_protocol(url.protocol().toString())
    , m_domain(url) { }

Site::Site(String&& protocol, RegistrableDomain&& domain)
    : m_protocol(WTFMove(protocol))
    , m_domain(WTFMove(domain)) { }

Site::Site(const SecurityOriginData& data)
    : m_protocol(data.protocol())
    , m_domain(data) { }

unsigned Site::hash() const
{
    return WTF::pairIntHash(m_protocol.hash(), m_domain.hash());
}

bool Site::matches(const URL& url) const
{
    return url.protocol() == m_protocol && m_domain.matches(url);
}

String Site::toString() const
{
    return isEmpty() ? emptyString() : makeString(m_protocol, "://"_s, m_domain.string());
}

TextStream& operator<<(TextStream& ts, const Site& site)
{
    ts << site.toString();
    return ts;
}

} // namespace WebKit
