/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 30, 2025.
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
#include "URLSearchParams.h"

#include "DOMURL.h"
#include <wtf/URLParser.h>

namespace WebCore {

URLSearchParams::URLSearchParams(const String& init, DOMURL* associatedURL)
    : m_associatedURL(associatedURL)
    , m_pairs(init.startsWith('?') ? WTF::URLParser::parseURLEncodedForm(StringView(init).substring(1)) : WTF::URLParser::parseURLEncodedForm(init))
{
}

URLSearchParams::URLSearchParams(const Vector<KeyValuePair<String, String>>& pairs)
    : m_pairs(pairs)
{
}

URLSearchParams::~URLSearchParams() = default;

ExceptionOr<Ref<URLSearchParams>> URLSearchParams::create(std::variant<Vector<Vector<String>>, Vector<KeyValuePair<String, String>>, String>&& variant)
{
    auto visitor = WTF::makeVisitor([&](const Vector<Vector<String>>& vector) -> ExceptionOr<Ref<URLSearchParams>> {
        Vector<KeyValuePair<String, String>> pairs;
        for (const auto& pair : vector) {
            if (pair.size() != 2)
                return Exception { ExceptionCode::TypeError };
            pairs.append({pair[0], pair[1]});
        }
        return adoptRef(*new URLSearchParams(WTFMove(pairs)));
    }, [&](const Vector<KeyValuePair<String, String>>& pairs) -> ExceptionOr<Ref<URLSearchParams>> {
        return adoptRef(*new URLSearchParams(pairs));
    }, [&](const String& string) -> ExceptionOr<Ref<URLSearchParams>> {
        return adoptRef(*new URLSearchParams(string, nullptr));
    });
    return std::visit(visitor, variant);
}

String URLSearchParams::get(const String& name) const
{
    for (const auto& pair : m_pairs) {
        if (pair.key == name)
            return pair.value;
    }
    return String();
}

bool URLSearchParams::has(const String& name, const String& value) const
{
    for (const auto& pair : m_pairs) {
        if (pair.key == name && (value.isNull() || pair.value == value))
            return true;
    }
    return false;
}

void URLSearchParams::sort()
{
    std::stable_sort(m_pairs.begin(), m_pairs.end(), [] (const auto& a, const auto& b) {
        return WTF::codePointCompareLessThan(a.key, b.key);
    });
    updateURL();
}

void URLSearchParams::set(const String& name, const String& value)
{
    for (auto& pair : m_pairs) {
        if (pair.key != name)
            continue;
        if (pair.value != value)
            pair.value = value;
        bool skippedFirstMatch = false;
        m_pairs.removeAllMatching([&] (const auto& pair) {
            if (pair.key == name) {
                if (skippedFirstMatch)
                    return true;
                skippedFirstMatch = true;
            }
            return false;
        });
        updateURL();
        return;
    }
    m_pairs.append({name, value});
    updateURL();
}

void URLSearchParams::append(const String& name, const String& value)
{
    m_pairs.append({name, value});
    updateURL();
}

Vector<String> URLSearchParams::getAll(const String& name) const
{
    return WTF::compactMap(m_pairs, [&](auto& pair) -> std::optional<String> {
        if (pair.key == name)
            return pair.value;
        return std::nullopt;
    });
}

void URLSearchParams::remove(const String& name, const String& value)
{
    m_pairs.removeAllMatching([&] (const auto& pair) {
        return pair.key == name && (value.isNull() || pair.value == value);
    });
    updateURL();
}

String URLSearchParams::toString() const
{
    return WTF::URLParser::serialize(m_pairs);
}

void URLSearchParams::updateURL()
{
    if (m_associatedURL)
        m_associatedURL->setSearch(WTF::URLParser::serialize(m_pairs));
}

void URLSearchParams::updateFromAssociatedURL()
{
    ASSERT(m_associatedURL);
    String search = m_associatedURL->search();
    m_pairs = search.startsWith('?') ? WTF::URLParser::parseURLEncodedForm(StringView(search).substring(1)) : WTF::URLParser::parseURLEncodedForm(search);
}

std::optional<KeyValuePair<String, String>> URLSearchParams::Iterator::next()
{
    auto& pairs = m_target->pairs();
    if (m_index >= pairs.size())
        return std::nullopt;

    auto& pair = pairs[m_index++];
    return KeyValuePair<String, String> { pair.key, pair.value };
}

URLSearchParams::Iterator::Iterator(URLSearchParams& params)
    : m_target(params)
{
}
    
}
