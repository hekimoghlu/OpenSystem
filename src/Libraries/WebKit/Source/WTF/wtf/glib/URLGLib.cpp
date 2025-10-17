/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 6, 2022.
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
#include "URL.h"

#if USE(GLIB)

#include <glib.h>
#include <wtf/URLParser.h>
#include <wtf/glib/GUniquePtr.h>
#include <wtf/text/CString.h>

namespace WTF {

URL::URL(GUri* uri)
{
    if (!uri) {
        invalidate();
        return;
    }

    GUniquePtr<char> uriString(g_uri_to_string(uri));
    URLParser parser(String::fromUTF8(uriString.get()));
    *this = parser.result();
}

GRefPtr<GUri> URL::createGUri() const
{
    if (isNull())
        return nullptr;

    return adoptGRef(g_uri_parse(m_string.utf8().data(),
        static_cast<GUriFlags>(G_URI_FLAGS_HAS_PASSWORD | G_URI_FLAGS_ENCODED_PATH | G_URI_FLAGS_ENCODED_QUERY | G_URI_FLAGS_ENCODED_FRAGMENT | G_URI_FLAGS_SCHEME_NORMALIZE | G_URI_FLAGS_PARSE_RELAXED),
        nullptr));
}

bool URL::hostIsIPAddress(StringView host)
{
    return !host.isEmpty() && g_hostname_is_ip_address(host.utf8().data());
}

} // namespace WTF

#endif // USE(GLIB)
