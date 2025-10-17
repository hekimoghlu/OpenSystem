/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 25, 2021.
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
#include "WebPasteboardProxy.h"

#include "Clipboard.h"
#include "Connection.h"
#include "SharedBufferReference.h"
#include "WebFrameProxy.h"
#include <WebCore/Pasteboard.h>
#include <WebCore/PasteboardCustomData.h>
#include <WebCore/PasteboardItemInfo.h>
#include <WebCore/PlatformPasteboard.h>
#include <WebCore/SelectionData.h>
#include <wtf/ListHashSet.h>
#include <wtf/SetForScope.h>

namespace WebKit {
using namespace WebCore;

void WebPasteboardProxy::getTypes(const String& pasteboardName, CompletionHandler<void(Vector<String>&&)>&& completionHandler)
{
    Clipboard::get(pasteboardName).formats(WTFMove(completionHandler));
}

void WebPasteboardProxy::readText(IPC::Connection& connection, const String& pasteboardName, CompletionHandler<void(String&&)>&& completionHandler)
{
    Clipboard::get(pasteboardName).readText(WTFMove(completionHandler), connection.inDispatchSyncMessageCount() > 1 ? Clipboard::ReadMode::Synchronous : Clipboard::ReadMode::Asynchronous);
}

void WebPasteboardProxy::readFilePaths(IPC::Connection& connection, const String& pasteboardName, CompletionHandler<void(Vector<String>&&)>&& completionHandler)
{
    Clipboard::get(pasteboardName).readFilePaths(WTFMove(completionHandler), connection.inDispatchSyncMessageCount() > 1 ? Clipboard::ReadMode::Synchronous : Clipboard::ReadMode::Asynchronous);
}

void WebPasteboardProxy::readBuffer(IPC::Connection& connection, const String& pasteboardName, const String& pasteboardType, CompletionHandler<void(RefPtr<SharedBuffer>&&)>&& completionHandler)
{
    Clipboard::get(pasteboardName).readBuffer(pasteboardType.utf8().data(), [completionHandler = WTFMove(completionHandler)](auto&& buffer) mutable {
        completionHandler(WTFMove(buffer));
    }, connection.inDispatchSyncMessageCount() > 1 ? Clipboard::ReadMode::Synchronous : Clipboard::ReadMode::Asynchronous);
}

void WebPasteboardProxy::writeToClipboard(const String& pasteboardName, SelectionData&& selectionData)
{
    Clipboard::get(pasteboardName).write(WTFMove(selectionData), [](int64_t) { });
}

void WebPasteboardProxy::clearClipboard(const String& pasteboardName)
{
    Clipboard::get(pasteboardName).clear();
}

void WebPasteboardProxy::setPrimarySelectionOwner(WebFrameProxy* frame)
{
    if (m_primarySelectionOwner == frame)
        return;

    if (m_primarySelectionOwner)
        m_primarySelectionOwner->collapseSelection();

    m_primarySelectionOwner = frame;
}

void WebPasteboardProxy::didDestroyFrame(WebFrameProxy* frame)
{
    if (frame == m_primarySelectionOwner)
        m_primarySelectionOwner = nullptr;
}

void WebPasteboardProxy::typesSafeForDOMToReadAndWrite(IPC::Connection& connection, const String& pasteboardName, const String& origin, std::optional<WebCore::PageIdentifier>, CompletionHandler<void(Vector<String>&&)>&& completionHandler)
{
    auto& clipboard = Clipboard::get(pasteboardName);
    clipboard.readBuffer(PasteboardCustomData::gtkType(), [&clipboard, origin, completionHandler = WTFMove(completionHandler)](auto&& buffer) mutable {
        ListHashSet<String> domTypes;
        auto customData = PasteboardCustomData::fromSharedBuffer(WTFMove(buffer));
        if (customData.origin() == origin) {
            for (auto& type : customData.orderedTypes())
                domTypes.add(type);
        }

        clipboard.formats([domTypes = WTFMove(domTypes), completionHandler = WTFMove(completionHandler)](Vector<String>&& formats) mutable {
            for (const auto& format : formats) {
                if (format == PasteboardCustomData::gtkType())
                    continue;

                if (Pasteboard::isSafeTypeForDOMToReadAndWrite(format))
                    domTypes.add(format);
            }

            completionHandler(copyToVector(domTypes));
        });
    }, connection.inDispatchSyncMessageCount() > 1 ? Clipboard::ReadMode::Synchronous : Clipboard::ReadMode::Asynchronous);
}

void WebPasteboardProxy::writeCustomData(IPC::Connection&, const Vector<PasteboardCustomData>& data, const String& pasteboardName, std::optional<WebCore::PageIdentifier>, CompletionHandler<void(int64_t)>&& completionHandler)
{
    auto& clipboard = Clipboard::get(pasteboardName);
    if (data.isEmpty() || data.size() > 1) {
        // We don't support more than one custom item in the clipboard.
        completionHandler(clipboard.changeCount());
        return;
    }

    SelectionData selectionData;
    const auto& customData = data[0];
    customData.forEachPlatformStringOrBuffer([&selectionData] (auto& type, auto& stringOrBuffer) {
        if (std::holds_alternative<Ref<SharedBuffer>>(stringOrBuffer)) {
            selectionData.addBuffer(type, std::get<Ref<SharedBuffer>>(stringOrBuffer));
        } else if (std::holds_alternative<String>(stringOrBuffer)) {
            if (type == "text/plain"_s)
                selectionData.setText(std::get<String>(stringOrBuffer));
            else if (type == "text/html"_s)
                selectionData.setMarkup(std::get<String>(stringOrBuffer));
            else if (type == "text/uri-list"_s)
                selectionData.setURIList(std::get<String>(stringOrBuffer));
        }
    });

    if (customData.hasSameOriginCustomData() || !customData.origin().isEmpty())
        selectionData.setCustomData(customData.createSharedBuffer());

    clipboard.write(WTFMove(selectionData), WTFMove(completionHandler));
}

static WebCore::PasteboardItemInfo pasteboardIemInfoFromFormats(Vector<String>&& formats)
{
    WebCore::PasteboardItemInfo info;
    if (formats.contains("text/plain"_s) || formats.contains("text/plain;charset=utf-8"_s))
        info.webSafeTypesByFidelity.append("text/plain"_s);
    if (formats.contains("text/html"_s))
        info.webSafeTypesByFidelity.append("text/html"_s);
    if (formats.contains("text/uri-list"_s))
        info.webSafeTypesByFidelity.append("text/uri-list"_s);
    if (formats.contains("image/png"_s))
        info.webSafeTypesByFidelity.append("image/png"_s);
    info.platformTypesByFidelity = WTFMove(formats);
    return info;
}

void WebPasteboardProxy::allPasteboardItemInfo(IPC::Connection&, const String& pasteboardName, int64_t changeCount, std::optional<WebCore::PageIdentifier>, CompletionHandler<void(std::optional<Vector<WebCore::PasteboardItemInfo>>&&)>&& completionHandler)
{
    auto& clipboard = Clipboard::get(pasteboardName);
    if (changeCount != clipboard.changeCount()) {
        completionHandler(std::nullopt);
        return;
    }

    clipboard.formats([completionHandler = WTFMove(completionHandler)](Vector<String>&& formats) mutable {
        completionHandler(Vector<WebCore::PasteboardItemInfo> { pasteboardIemInfoFromFormats(WTFMove(formats)) });
    });
}

void WebPasteboardProxy::informationForItemAtIndex(IPC::Connection&, size_t index, const String& pasteboardName, int64_t changeCount, std::optional<WebCore::PageIdentifier>, CompletionHandler<void(std::optional<WebCore::PasteboardItemInfo>&&)>&& completionHandler)
{
    auto& clipboard = Clipboard::get(pasteboardName);
    if (changeCount != clipboard.changeCount() || index) {
        completionHandler(std::nullopt);
        return;
    }

    clipboard.formats([completionHandler = WTFMove(completionHandler)](Vector<String>&& formats) mutable {
        completionHandler(pasteboardIemInfoFromFormats(WTFMove(formats)));
    });
}

void WebPasteboardProxy::getPasteboardItemsCount(IPC::Connection&, const String& pasteboardName, std::optional<WebCore::PageIdentifier>, CompletionHandler<void(uint64_t)>&& completionHandler)
{
    Clipboard::get(pasteboardName).formats([completionHandler = WTFMove(completionHandler)](Vector<String>&& formats) mutable {
        completionHandler(formats.isEmpty() ? 0 : 1);
    });
}

void WebPasteboardProxy::readURLFromPasteboard(IPC::Connection& connection, size_t index, const String& pasteboardName, std::optional<WebCore::PageIdentifier>, CompletionHandler<void(String&& url, String&& title)>&& completionHandler)
{
    if (index) {
        completionHandler({ }, { });
        return;
    }

    Clipboard::get(pasteboardName).readURL(WTFMove(completionHandler), connection.inDispatchSyncMessageCount() > 1 ? Clipboard::ReadMode::Synchronous : Clipboard::ReadMode::Asynchronous);
}

void WebPasteboardProxy::readBufferFromPasteboard(IPC::Connection& connection, std::optional<size_t> index, const String& pasteboardType, const String& pasteboardName, std::optional<WebCore::PageIdentifier>, CompletionHandler<void(RefPtr<WebCore::SharedBuffer>&&)>&& completionHandler)
{
    if (index && index.value()) {
        completionHandler({ });
        return;
    }

    Clipboard::get(pasteboardName).readBuffer(pasteboardType.utf8().data(), [completionHandler = WTFMove(completionHandler)](auto&& buffer) mutable {
        completionHandler(WTFMove(buffer));
    }, connection.inDispatchSyncMessageCount() > 1 ? Clipboard::ReadMode::Synchronous : Clipboard::ReadMode::Asynchronous);
}

void WebPasteboardProxy::getPasteboardChangeCount(IPC::Connection&, const String& pasteboardName, CompletionHandler<void(int64_t)>&& completionHandler)
{
    completionHandler(Clipboard::get(pasteboardName).changeCount());
}

} // namespace WebKit
