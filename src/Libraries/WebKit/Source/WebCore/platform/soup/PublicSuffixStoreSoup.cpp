/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 25, 2022.
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

#include <libsoup/soup.h>
#include <wtf/glib/GUniquePtr.h>

namespace WebCore {

bool PublicSuffixStore::platformIsPublicSuffix(StringView domain) const
{
    if (domain.isEmpty())
        return false;

    return soup_tld_domain_is_public_suffix(domain.convertToASCIILowercase().utf8().data());
}

static String permissiveTopPrivateDomain(StringView domain)
{
    auto position = domain.length();
    bool foundDot = false;

    /* If a domain does not have a known public suffix we can just assume
     * the last pair of segments is probably the suffix.
     * Currently this is only used for web-platform.test. */
    while (position-- > 0) {
        if (domain[position] == '.') {
            if (foundDot)
                return domain.substring(position + 1).toString();
            foundDot = true;
        }
    }

    return foundDot ? domain.toString() : String();
}

String PublicSuffixStore::platformTopPrivatelyControlledDomain(StringView domain) const
{
    // This function is expected to work with the format used by cookies, so skip any leading dots.
    unsigned position = 0;
    while (domain[position] == '.')
        position++;

    if (position == domain.length())
        return String();

    auto tldView = domain.substring(position);
    const auto tldCString = tldView.utf8();

    GUniqueOutPtr<GError> error;
    if (const char* baseDomain = soup_tld_get_base_domain(tldCString.data(), &error.outPtr()))
        return String::fromUTF8(baseDomain);

    if (g_error_matches(error.get(), SOUP_TLD_ERROR, SOUP_TLD_ERROR_NO_BASE_DOMAIN)) {
        if (domain.endsWithIgnoringASCIICase("web-platform.test"_s))
            return permissiveTopPrivateDomain(tldView);
        return String();
    }

    if (g_error_matches(error.get(), SOUP_TLD_ERROR, SOUP_TLD_ERROR_INVALID_HOSTNAME) || g_error_matches(error.get(), SOUP_TLD_ERROR, SOUP_TLD_ERROR_NOT_ENOUGH_DOMAINS))
        return String();

    if (g_error_matches(error.get(), SOUP_TLD_ERROR, SOUP_TLD_ERROR_IS_IP_ADDRESS))
        return domain.toString();

    ASSERT_NOT_REACHED();
    return String();
}

} // namespace WebCore
