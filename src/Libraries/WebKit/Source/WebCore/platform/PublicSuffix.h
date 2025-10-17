/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 23, 2023.
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

#include <wtf/CrossThreadCopier.h>
#include <wtf/HashTraits.h>
#include <wtf/text/StringHash.h>

namespace WebCore {

class PublicSuffix {
public:
    static PublicSuffix fromRawString(String&& string) { return PublicSuffix(WTFMove(string)); }
    PublicSuffix() = default;
    bool isValid() const { return !m_string.isEmpty(); }
    const String& string() const { return m_string; }
    PublicSuffix isolatedCopy() const { return fromRawString(crossThreadCopy(m_string)); }

    PublicSuffix(WTF::HashTableDeletedValueType)
        : m_string(WTF::HashTableDeletedValue) { }
    friend bool operator==(const PublicSuffix&, const PublicSuffix&) = default;
    bool operator==(ASCIILiteral other) const { return m_string == other; }
    bool isHashTableDeletedValue() const { return m_string.isHashTableDeletedValue(); }
    unsigned hash() const { return m_string.hash(); }
    struct PublicSuffixHash {
        static unsigned hash(const PublicSuffix& publicSuffix) { return ASCIICaseInsensitiveHash::hash(publicSuffix.m_string.impl()); }
        static bool equal(const PublicSuffix& a, const PublicSuffix& b) { return equalIgnoringASCIICase(a.string(), b.string()); }
        static const bool safeToCompareToEmptyOrDeleted = false;
    };

private:
    explicit PublicSuffix(String&& string) : m_string(WTFMove(string)) { }

    String m_string;
};

} // namespace WebCore

namespace WTF {

template<> struct DefaultHash<WebCore::PublicSuffix> : WebCore::PublicSuffix::PublicSuffixHash { };
template<> struct HashTraits<WebCore::PublicSuffix> : SimpleClassHashTraits<WebCore::PublicSuffix> { };

} // namespace WTF
