/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 29, 2023.
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
#include "HTTPHeaderMap.h"

#include <utility>
#include <wtf/CrossThreadCopier.h>
#include <wtf/text/MakeString.h>
#include <wtf/text/StringView.h>

#if USE(CF)
#include <wtf/cf/VectorCF.h>
#endif

namespace WebCore {

HTTPHeaderMap::HTTPHeaderMap() = default;

HTTPHeaderMap::HTTPHeaderMap(CommonHeadersVector&& commonHeaders, UncommonHeadersVector&& uncommonHeaders)
    : m_commonHeaders(WTFMove(commonHeaders))
    , m_uncommonHeaders(WTFMove(uncommonHeaders))
{
}

HTTPHeaderMap HTTPHeaderMap::isolatedCopy() const &
{
    HTTPHeaderMap map;
    map.m_commonHeaders = crossThreadCopy(m_commonHeaders);
    map.m_uncommonHeaders = crossThreadCopy(m_uncommonHeaders);
    return map;
}

HTTPHeaderMap HTTPHeaderMap::isolatedCopy() &&
{
    HTTPHeaderMap map;
    map.m_commonHeaders = crossThreadCopy(WTFMove(m_commonHeaders));
    map.m_uncommonHeaders = crossThreadCopy(WTFMove(m_uncommonHeaders));
    return map;
}

String HTTPHeaderMap::get(StringView name) const
{
    HTTPHeaderName headerName;
    if (findHTTPHeaderName(name, headerName))
        return get(headerName);

    return getUncommonHeader(name);
}

String HTTPHeaderMap::getUncommonHeader(StringView name) const
{
    auto index = m_uncommonHeaders.findIf([&](auto& header) {
        return equalIgnoringASCIICase(header.key, name);
    });
    return index != notFound ? m_uncommonHeaders[index].value : String();
}

#if USE(CF)

void HTTPHeaderMap::set(CFStringRef name, const String& value)
{
    // Fast path: avoid constructing a temporary String in the common header case.
    if (auto asciiCharacters = CFStringGetASCIICStringSpan(name); asciiCharacters.data()) {
        HTTPHeaderName headerName;
        if (findHTTPHeaderName(StringView(asciiCharacters), headerName))
            set(headerName, value);
        else
            setUncommonHeader(String(asciiCharacters), value);

        return;
    }

    set(String(name), value);
}

#endif // USE(CF)

void HTTPHeaderMap::set(const String& name, const String& value)
{
    HTTPHeaderName headerName;
    if (findHTTPHeaderName(name, headerName)) {
        set(headerName, value);
        return;
    }

    setUncommonHeader(name, value);
}

void HTTPHeaderMap::setUncommonHeader(const String& name, const String& value)
{
#if ASSERT_ENABLED
    HTTPHeaderName headerName;
    ASSERT(!findHTTPHeaderName(name, headerName));
#endif

    auto index = m_uncommonHeaders.findIf([&](auto& header) {
        return equalIgnoringASCIICase(header.key, name);
    });
    if (index == notFound)
        m_uncommonHeaders.append(UncommonHeader { name, value });
    else
        m_uncommonHeaders[index].value = value;
}

void HTTPHeaderMap::add(const String& name, const String& value)
{
    HTTPHeaderName headerName;
    if (findHTTPHeaderName(name, headerName)) {
        add(headerName, value);
        return;
    }
    addUncommonHeader(name, value);
}

void HTTPHeaderMap::addUncommonHeader(const String& name, const String& value)
{
#if ASSERT_ENABLED
    HTTPHeaderName headerName;
    ASSERT(!findHTTPHeaderName(name, headerName));
#endif

    auto index = m_uncommonHeaders.findIf([&](auto& header) {
        return equalIgnoringASCIICase(header.key, name);
    });
    if (index == notFound)
        m_uncommonHeaders.append(UncommonHeader { name, value });
    else
        m_uncommonHeaders[index].value = makeString(m_uncommonHeaders[index].value, ", "_s, value);
}

void HTTPHeaderMap::append(const String& name, const String& value)
{
    ASSERT(!contains(name));

    HTTPHeaderName headerName;
    if (findHTTPHeaderName(name, headerName))
        m_commonHeaders.append(CommonHeader { headerName, value });
    else
        m_uncommonHeaders.append(UncommonHeader { name, value });
}

bool HTTPHeaderMap::addIfNotPresent(HTTPHeaderName headerName, const String& value)
{
    if (contains(headerName))
        return false;

    m_commonHeaders.append(CommonHeader { headerName, value });
    return true;
}

bool HTTPHeaderMap::contains(const String& name) const
{
    HTTPHeaderName headerName;
    if (findHTTPHeaderName(name, headerName))
        return contains(headerName);

    return m_uncommonHeaders.findIf([&](auto& header) {
        return equalIgnoringASCIICase(header.key, name);
    }) != notFound;
}

bool HTTPHeaderMap::remove(const String& name)
{
    HTTPHeaderName headerName;
    if (findHTTPHeaderName(name, headerName))
        return remove(headerName);

    return m_uncommonHeaders.removeFirstMatching([&](auto& header) {
        return equalIgnoringASCIICase(header.key, name);
    });
}

String HTTPHeaderMap::get(HTTPHeaderName name) const
{
    auto index = m_commonHeaders.findIf([&](auto& header) {
        return header.key == name;
    });
    return index != notFound ? m_commonHeaders[index].value : String();
}

void HTTPHeaderMap::set(HTTPHeaderName name, const String& value)
{
    auto index = m_commonHeaders.findIf([&](auto& header) {
        return header.key == name;
    });
    if (index == notFound)
        m_commonHeaders.append(CommonHeader { name, value });
    else
        m_commonHeaders[index].value = value;
}

bool HTTPHeaderMap::contains(HTTPHeaderName name) const
{
    return m_commonHeaders.findIf([&](auto& header) {
        return header.key == name;
    }) != notFound;
}

bool HTTPHeaderMap::remove(HTTPHeaderName name)
{
    return m_commonHeaders.removeFirstMatching([&](auto& header) {
        return header.key == name;
    });
}

void HTTPHeaderMap::add(HTTPHeaderName name, const String& value)
{
    auto index = m_commonHeaders.findIf([&](auto& header) {
        return header.key == name;
    });
    if (index != notFound)
        m_commonHeaders[index].value = makeString(m_commonHeaders[index].value, ", "_s, value);
    else
        m_commonHeaders.append(CommonHeader { name, value });
}

} // namespace WebCore
