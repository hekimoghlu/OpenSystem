/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 9, 2022.
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
#include "ContentSecurityPolicySourceListDirective.h"

#include "ContentSecurityPolicy.h"
#include "ContentSecurityPolicyDirectiveList.h"
#include <wtf/TZoneMallocInlines.h>
#include <wtf/URL.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(ContentSecurityPolicySourceListDirective);

ContentSecurityPolicySourceListDirective::ContentSecurityPolicySourceListDirective(const ContentSecurityPolicyDirectiveList& directiveList, const String& name, const String& value)
    : ContentSecurityPolicyDirective(directiveList, name, value)
    , m_sourceList(directiveList.policy(), name)
{
    m_sourceList.parse(value);
}

bool ContentSecurityPolicySourceListDirective::allows(const URL& url, bool didReceiveRedirectResponse, ShouldAllowEmptyURLIfSourceListIsNotNone shouldAllowEmptyURLIfSourceListEmpty)
{
    if (url.isEmpty())
        return shouldAllowEmptyURLIfSourceListEmpty == ShouldAllowEmptyURLIfSourceListIsNotNone::Yes && !m_sourceList.isNone();
    return m_sourceList.matches(url, didReceiveRedirectResponse);
}

bool ContentSecurityPolicySourceListDirective::allows(const String& nonce) const
{
    return m_sourceList.matches(nonce);
}

bool ContentSecurityPolicySourceListDirective::containsAllHashes(const Vector<ContentSecurityPolicyHash>& hashes) const
{
    return m_sourceList.matchesAll(hashes);
}

bool ContentSecurityPolicySourceListDirective::allows(const Vector<ContentSecurityPolicyHash>& hashes) const
{
    return m_sourceList.matches(hashes);
}

bool ContentSecurityPolicySourceListDirective::allowUnsafeHashes(const Vector<ContentSecurityPolicyHash>& hashes) const
{
    return m_sourceList.allowUnsafeHashes() && allows(hashes);
}

} // namespace WebCore
