/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 9, 2024.
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
#include "PublicSuffixStore.h"

#include <libpsl.h>
#include <wtf/URL.h>

namespace WebCore {

bool PublicSuffixStore::platformIsPublicSuffix(StringView domain) const
{
    if (domain.isEmpty())
        return false;

    const psl_ctx_t* psl = psl_builtin();
    ASSERT(psl);
    bool ret = psl_is_public_suffix2(psl, domain.toStringWithoutCopying().convertToLowercaseWithoutLocale().utf8().data(), PSL_TYPE_ANY | PSL_TYPE_NO_STAR_RULE);
    return ret;
}

static String topPrivatelyControlledDomainInternal(const psl_ctx_t* psl, const char* domain)
{
    // psl_registerable_domain returns a pointer to domain's data or null if there is no private domain
    if (const char* topPrivateDomain = psl_registrable_domain(psl, domain))
        return String::fromLatin1(topPrivateDomain);
    return String();
}

String PublicSuffixStore::platformTopPrivatelyControlledDomain(StringView domain) const
{
    if (platformIsPublicSuffix(domain))
        return String();

    // This function is expected to work with the format used by cookies, so skip any leading dots.
    auto domainUTF8 = domain.utf8();

    unsigned position = 0;
    while (domainUTF8.data()[position] == '.')
        position++;

    if (position == domainUTF8.length())
        return String();

    const psl_ctx_t* psl = psl_builtin();
    ASSERT(psl);
    return topPrivatelyControlledDomainInternal(psl, domainUTF8.data() + position);
}

} // namespace WebCore
