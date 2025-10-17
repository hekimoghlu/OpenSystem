/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 3, 2022.
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
#include "Pasteboard.h"

#include "CommonAtomStrings.h"
#include "PasteboardStrategy.h"
#include "PlatformStrategies.h"
#include "Settings.h"
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/StringHash.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(Pasteboard);

bool Pasteboard::isSafeTypeForDOMToReadAndWrite(const String& type)
{
    return type == textPlainContentTypeAtom() || type == textHTMLContentTypeAtom() || type == "text/uri-list"_s;
}

bool Pasteboard::canExposeURLToDOMWhenPasteboardContainsFiles(const String& urlString)
{
    URL url({ }, urlString);
    return url.protocolIsInHTTPFamily() || url.protocolIsBlob() || url.protocolIsData();
}

#if !PLATFORM(COCOA)

Vector<String> Pasteboard::readAllStrings(const String& type)
{
    auto result = readString(type);
    if (result.isEmpty())
        return { };

    return { result };
}

#endif

std::optional<Vector<PasteboardItemInfo>> Pasteboard::allPasteboardItemInfo() const
{
#if PLATFORM(COCOA) || PLATFORM(GTK)
    if (auto* strategy = platformStrategies()->pasteboardStrategy())
        return strategy->allPasteboardItemInfo(name(), m_changeCount, context());
#endif
    return std::nullopt;
}

std::optional<PasteboardItemInfo> Pasteboard::pasteboardItemInfo(size_t index) const
{
#if PLATFORM(COCOA) || PLATFORM(GTK)
    if (auto* strategy = platformStrategies()->pasteboardStrategy())
        return strategy->informationForItemAtIndex(index, name(), m_changeCount, context());
#else
    UNUSED_PARAM(index);
#endif
    return std::nullopt;
}

String Pasteboard::readString(size_t index, const String& type)
{
    if (auto* strategy = platformStrategies()->pasteboardStrategy())
        return strategy->readStringFromPasteboard(index, type, name(), context());
    return { };
}

RefPtr<WebCore::SharedBuffer> Pasteboard::readBuffer(std::optional<size_t> index, const String& type)
{
    if (auto* strategy = platformStrategies()->pasteboardStrategy())
        return strategy->readBufferFromPasteboard(index, type, name(), context());
    return nullptr;
}

URL Pasteboard::readURL(size_t index, String& title)
{
    if (auto* strategy = platformStrategies()->pasteboardStrategy())
        return strategy->readURLFromPasteboard(index, name(), title, context());
    return { };
}

#if !PLATFORM(MAC)

bool Pasteboard::canWriteTrustworthyWebURLsPboardType()
{
    return false;
}

#endif

};
