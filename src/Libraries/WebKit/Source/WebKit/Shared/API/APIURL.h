/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 14, 2024.
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
#include <wtf/Forward.h>
#include <wtf/URL.h>
#include <wtf/text/WTFString.h>

namespace API {

class URL : public ObjectImpl<Object::Type::URL> {
public:
    static Ref<API::URL> create(const WTF::String& string)
    {
        return adoptRef(*new URL(string));
    }

    static Ref<API::URL> create(const API::URL* baseURL, const WTF::String& relativeURL)
    {
        ASSERT(baseURL);
        baseURL->parseURLIfNecessary();
        auto absoluteURL = makeUnique<WTF::URL>(*baseURL->m_parsedURL.get(), relativeURL);
        const WTF::String& absoluteURLString = absoluteURL->string();

        return adoptRef(*new API::URL(WTFMove(absoluteURL), absoluteURLString));
    }

    bool isNull() const { return m_string.isNull(); }
    bool isEmpty() const { return m_string.isEmpty(); }

    const WTF::String& string() const { return m_string; }

    static bool equals(const API::URL& a, const API::URL& b)
    {
        return a.url() == b.url();
    }

    WTF::String host() const
    {
        parseURLIfNecessary();
        return m_parsedURL->host().toString();
    }

    WTF::String protocol() const
    {
        parseURLIfNecessary();
        return m_parsedURL->protocol().toString();
    }

    WTF::String path() const
    {
        parseURLIfNecessary();
        return m_parsedURL->path().toString();
    }

    WTF::String lastPathComponent() const
    {
        parseURLIfNecessary();
        return m_parsedURL->lastPathComponent().toString();
    }

private:
    URL(const WTF::String& string)
        : m_string(string)
    {
    }

    URL(std::unique_ptr<WTF::URL> parsedURL, const WTF::String& string)
        : m_string(string)
        , m_parsedURL(WTFMove(parsedURL))
    {
    }

    const WTF::URL& url() const
    {
        parseURLIfNecessary();
        return *m_parsedURL;
    }

    void parseURLIfNecessary() const
    {
        if (m_parsedURL)
            return;
        m_parsedURL = makeUnique<WTF::URL>(m_string);
    }

    WTF::String m_string;
    mutable std::unique_ptr<WTF::URL> m_parsedURL;
};

} // namespace WebKit

SPECIALIZE_TYPE_TRAITS_API_OBJECT(URL);
