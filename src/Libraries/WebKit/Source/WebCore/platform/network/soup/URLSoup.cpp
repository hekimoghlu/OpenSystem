/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 21, 2023.
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
#include "URLSoup.h"

#include <wtf/URL.h>

namespace WebCore {

#if USE(SOUP2)
URL soupURIToURL(SoupURI* soupURI)
{
    if (!soupURI)
        return URL();

    GUniquePtr<gchar> urlString(soup_uri_to_string(soupURI, FALSE));
    URL url { String::fromUTF8(urlString.get()) };
    if (url.isValid()) {
        // Motivated by https://bugs.webkit.org/show_bug.cgi?id=38956. libsoup
        // does not add the password to the URL when calling
        // soup_uri_to_string, and thus the requests are not properly
        // built. Fixing soup_uri_to_string is a no-no as the maintainer does
        // not want to break compatibility with previous implementations
        if (soupURI->password)
            url.setPassword(String::fromUTF8(soupURI->password));
    }

    return url;
}

GUniquePtr<SoupURI> urlToSoupURI(const URL& url)
{
    if (!url.isValid())
        return nullptr;

    return GUniquePtr<SoupURI>(soup_uri_new(url.string().utf8().data()));
}

#else // !USE(SOUP2)

URL soupURIToURL(GUri* uri)
{
    return uri;
}

GRefPtr<GUri> urlToSoupURI(const URL& url)
{
    return url.createGUri();
}
#endif // USE(SOUP2)

} // namespace WebCore
