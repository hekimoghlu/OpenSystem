/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 21, 2022.
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

#include <wtf/Forward.h>
#include <wtf/URL.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class UserContentURLPattern {
public:
    UserContentURLPattern() = default;

    enum class Error : uint8_t {
        None,
        Invalid,
        MissingScheme,
        MissingHost,
        InvalidHost,
        MissingPath,
    };

    explicit UserContentURLPattern(StringView pattern)
    {
        m_error = parse(pattern);
    }

    WEBCORE_EXPORT UserContentURLPattern(StringView scheme, StringView host, StringView path);

    bool isValid() const { return m_error == Error::None; }
    Error error() const { return m_error; }

    template <typename T>
    bool matches(const T& test) const
    {
        if (!isValid())
            return false;
        return matchesScheme(test) && matchesHost(test) && matchesPath(test);
    }

    const String& scheme() const { return m_scheme; }
    const String& host() const { return m_host; }
    const String& path() const { return m_path; }

    bool matchAllHosts() const { return m_matchSubdomains && m_host.isEmpty(); }
    bool matchSubdomains() const { return m_matchSubdomains; }

    // The host with the '*' wildcard, if matchSubdomains is true, otherwise same as host().
    WEBCORE_EXPORT String originalHost() const;

    WEBCORE_EXPORT bool matchesScheme(const URL&) const;
    bool matchesHost(const URL& url) const { return matchesHost(url.host().toStringWithoutCopying()); }
    bool matchesPath(const URL& url) const { return matchesPath(url.path().toStringWithoutCopying()); }

    WEBCORE_EXPORT bool matchesScheme(const UserContentURLPattern&) const;
    bool matchesHost(const UserContentURLPattern& other) const { return matchesHost(other.host()); }
    bool matchesPath(const UserContentURLPattern& other) const { return matchesPath(other.path()); }

    WEBCORE_EXPORT bool operator==(const UserContentURLPattern& other) const;

    static bool matchesPatterns(const URL&, const Vector<String>& allowlist, const Vector<String>& blocklist);

private:
    WEBCORE_EXPORT Error parse(StringView pattern);
    void normalizeHostAndSetMatchSubdomains();

    WEBCORE_EXPORT bool matchesHost(const String&) const;
    WEBCORE_EXPORT bool matchesPath(const String&) const;

    String m_scheme;
    String m_host;
    String m_path;

    Error m_error { Error::Invalid };
    bool m_matchSubdomains { false };
};

WEBCORE_EXPORT bool matchesWildcardPattern(const String& pattern, const String& testString);

} // namespace WebCore
