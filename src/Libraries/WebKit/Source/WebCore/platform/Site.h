/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 8, 2021.
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

#include "RegistrableDomain.h"
#include <wtf/HashTraits.h>

namespace WebCore {

// https://html.spec.whatwg.org/multipage/browsers.html#site
class Site {
public:
    WEBCORE_EXPORT explicit Site(const URL&);
    WEBCORE_EXPORT explicit Site(String&& protocol, RegistrableDomain&&);
    WEBCORE_EXPORT explicit Site(const SecurityOriginData&);

    Site(const Site&) = default;
    Site& operator=(const Site&) = default;

    const String& protocol() const { return m_protocol; }
    const RegistrableDomain& domain() const { return m_domain; }
    WEBCORE_EXPORT String toString() const;
    bool isEmpty() const { return m_domain.isEmpty(); }
    WEBCORE_EXPORT bool matches(const URL&) const;

    Site(WTF::HashTableEmptyValueType) { }
    Site(WTF::HashTableDeletedValueType deleted)
        : m_protocol(deleted) { }
    bool isHashTableDeletedValue() const { return m_protocol.isHashTableDeletedValue(); }
    WEBCORE_EXPORT unsigned hash() const;

    bool operator==(const Site&) const = default;
    bool operator!=(const Site&) const = default;

    struct Hash {
        static unsigned hash(const Site& site) { return site.hash(); }
        static bool equal(const Site& a, const Site& b) { return a == b; }
        static const bool safeToCompareToEmptyOrDeleted = false;
    };

private:
    String m_protocol;
    RegistrableDomain m_domain;
};

WEBCORE_EXPORT TextStream& operator<<(TextStream&, const Site&);

} // namespace WebCore

namespace WTF {
template<> struct DefaultHash<WebCore::Site> : WebCore::Site::Hash { };
template<> struct HashTraits<WebCore::Site> : SimpleClassHashTraits<WebCore::Site> {
    static WebCore::Site emptyValue() { return { WTF::HashTableEmptyValue }; }
};
}
