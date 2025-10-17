/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 6, 2025.
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
#include "WebPasteboardOverrides.h"

#include <WebCore/PasteboardItemInfo.h>
#include <wtf/NeverDestroyed.h>

namespace WebKit {
using namespace WebCore;

WebPasteboardOverrides& WebPasteboardOverrides::sharedPasteboardOverrides()
{
    static NeverDestroyed<WebPasteboardOverrides> sharedOverrides;
    return sharedOverrides;
}

WebPasteboardOverrides::WebPasteboardOverrides()
{
}

void WebPasteboardOverrides::addOverride(const String& pasteboardName, const String& type, const Vector<uint8_t>& data)
{
    auto& overrides = m_overridesMap.add(pasteboardName, HashMap<String, Vector<uint8_t>>()).iterator->value;
    overrides.set(type, data);
}

void WebPasteboardOverrides::removeOverride(const String& pasteboardName, const String& type)
{
    auto it = m_overridesMap.find(pasteboardName);
    if (it == m_overridesMap.end())
        return;

    it->value.remove(type);

    // If this was the last override for this pasteboard, remove its record completely.
    if (it->value.isEmpty())
        m_overridesMap.remove(it);
}

Vector<String> WebPasteboardOverrides::overriddenTypes(const String& pasteboardName)
{
    auto it = m_overridesMap.find(pasteboardName);
    if (it == m_overridesMap.end())
        return { };

    return copyToVector(it->value.keys());
}

std::optional<WebCore::PasteboardItemInfo> WebPasteboardOverrides::overriddenInfo(const String& pasteboardName)
{
    auto types = this->overriddenTypes(pasteboardName);
    if (types.isEmpty())
        return std::nullopt;

    PasteboardItemInfo item;
    item.platformTypesByFidelity = types;
    // FIXME: This is currently appropriate for all clients that rely on PasteboardItemInfo, but we may need to adjust
    // this in the future so that we don't treat 'inline' types such as plain text as uploaded files.
    item.platformTypesForFileUpload = types;
    return { WTFMove(item) };
}

bool WebPasteboardOverrides::getDataForOverride(const String& pasteboardName, const String& type, Vector<uint8_t>& data) const
{
    auto pasteboardIterator = m_overridesMap.find(pasteboardName);
    if (pasteboardIterator == m_overridesMap.end())
        return false;

    auto typeIterator = pasteboardIterator->value.find(type);
    if (typeIterator == pasteboardIterator->value.end())
        return false;

    data = typeIterator->value;
    return true;
}

} // namespace WebKit
