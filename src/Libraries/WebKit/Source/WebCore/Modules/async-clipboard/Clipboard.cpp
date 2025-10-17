/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 6, 2022.
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
#include "Clipboard.h"

#include "ClipboardImageReader.h"
#include "ClipboardItem.h"
#include "CommonAtomStrings.h"
#include "Document.h"
#include "Editor.h"
#include "JSBlob.h"
#include "JSClipboardItem.h"
#include "JSDOMPromiseDeferred.h"
#include "LocalFrame.h"
#include "Navigator.h"
#include "Page.h"
#include "PagePasteboardContext.h"
#include "Pasteboard.h"
#include "Settings.h"
#include "SharedBuffer.h"
#include "UserGestureIndicator.h"
#include "WebContentReader.h"
#include <wtf/CompletionHandler.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(Clipboard);

static bool shouldProceedWithClipboardWrite(const LocalFrame& frame)
{
    auto& settings = frame.settings();
    if (settings.javaScriptCanAccessClipboard() || frame.editor().isCopyingFromMenuOrKeyBinding())
        return true;

    switch (settings.clipboardAccessPolicy()) {
    case ClipboardAccessPolicy::Allow:
        return true;
    case ClipboardAccessPolicy::RequiresUserGesture:
        return UserGestureIndicator::processingUserGesture();
    case ClipboardAccessPolicy::Deny:
        return false;
    }

    ASSERT_NOT_REACHED();
    return false;
}

Ref<Clipboard> Clipboard::create(Navigator& navigator)
{
    return adoptRef(*new Clipboard(navigator));
}

Clipboard::Clipboard(Navigator& navigator)
    : m_navigator(navigator)
{
}

Clipboard::~Clipboard()
{
    if (auto writer = WTFMove(m_activeItemWriter))
        writer->invalidate();
}

Navigator* Clipboard::navigator()
{
    return m_navigator.get();
}

enum EventTargetInterfaceType Clipboard::eventTargetInterface() const
{
    return EventTargetInterfaceType::Clipboard;
}

ScriptExecutionContext* Clipboard::scriptExecutionContext() const
{
    return m_navigator ? m_navigator->scriptExecutionContext() : nullptr;
}

void Clipboard::readText(Ref<DeferredPromise>&& promise)
{
    RefPtr frame = this->frame();
    if (!frame) {
        promise->reject(ExceptionCode::NotAllowedError);
        return;
    }

    auto pasteboard = Pasteboard::createForCopyAndPaste(PagePasteboardContext::create(frame->pageID()));
    auto changeCountAtStart = pasteboard->changeCount();
    if (!frame->requestDOMPasteAccess()) {
        promise->reject(ExceptionCode::NotAllowedError);
        return;
    }

    auto allInfo = pasteboard->allPasteboardItemInfo();
    if (!allInfo) {
        promise->reject(ExceptionCode::NotAllowedError);
        return;
    }

    String text;
    for (size_t index = 0; index < allInfo->size(); ++index) {
        if (allInfo->at(index).webSafeTypesByFidelity.contains(textPlainContentTypeAtom())) {
            PasteboardPlainText plainTextReader;
            pasteboard->read(plainTextReader, PlainTextURLReadingPolicy::IgnoreURL, index);
            text = WTFMove(plainTextReader.text);
            break;
        }
    }

    if (changeCountAtStart == pasteboard->changeCount())
        promise->resolve<IDLDOMString>(WTFMove(text));
    else
        promise->reject(ExceptionCode::NotAllowedError);
}

void Clipboard::writeText(const String& data, Ref<DeferredPromise>&& promise)
{
    RefPtr frame = this->frame();
    RefPtr document = frame ? frame->document() : nullptr;
    if (!document || !shouldProceedWithClipboardWrite(*frame)) {
        promise->reject(ExceptionCode::NotAllowedError);
        return;
    }

    PasteboardCustomData customData;
    customData.writeString(textPlainContentTypeAtom(), data);
    customData.setOrigin(document->originIdentifierForPasteboard());
    Pasteboard::createForCopyAndPaste(PagePasteboardContext::create(frame->pageID()))->writeCustomData({ WTFMove(customData) });
    promise->resolve();
}

void Clipboard::read(Ref<DeferredPromise>&& promise)
{
    auto rejectPromiseAndClearActiveSession = [&] {
        m_activeSession = std::nullopt;
        promise->reject(ExceptionCode::NotAllowedError);
    };

    RefPtr frame = this->frame();
    if (!frame) {
        rejectPromiseAndClearActiveSession();
        return;
    }

    auto pasteboard = Pasteboard::createForCopyAndPaste(PagePasteboardContext::create(frame->pageID()));
    auto changeCountAtStart = pasteboard->changeCount();

    if (!frame->requestDOMPasteAccess()) {
        rejectPromiseAndClearActiveSession();
        return;
    }

    if (!m_activeSession || m_activeSession->changeCount != changeCountAtStart) {
        auto allInfo = pasteboard->allPasteboardItemInfo();
        if (!allInfo) {
            rejectPromiseAndClearActiveSession();
            return;
        }

        auto clipboardItems = allInfo->map([this](auto& itemInfo) {
            return ClipboardItem::create(*this, itemInfo);
        });
        m_activeSession = {{ WTFMove(pasteboard), WTFMove(clipboardItems), changeCountAtStart }};
    }

    promise->resolve<IDLSequence<IDLInterface<ClipboardItem>>>(m_activeSession->items);
}

void Clipboard::getType(ClipboardItem& item, const String& type, Ref<DeferredPromise>&& promise)
{
    if (!m_activeSession) {
        promise->reject(ExceptionCode::NotAllowedError);
        return;
    }

    RefPtr frame = this->frame();
    if (!frame) {
        m_activeSession = std::nullopt;
        promise->reject(ExceptionCode::NotAllowedError);
        return;
    }

    auto itemIndex = m_activeSession->items.findIf([&] (auto& activeItem) {
        return activeItem.ptr() == &item;
    });

    if (itemIndex == notFound) {
        promise->reject(ExceptionCode::NotAllowedError);
        return;
    }

    if (!item.types().contains(type)) {
        promise->reject(ExceptionCode::NotAllowedError);
        return;
    }

    if (type == "image/png"_s) {
        ClipboardImageReader imageReader { frame->document(), type };
        activePasteboard().read(imageReader, itemIndex);
        auto imageBlob = imageReader.takeResult();
        if (updateSessionValidity() == SessionIsValid::Yes && imageBlob)
            promise->resolve<IDLInterface<Blob>>(imageBlob.releaseNonNull());
        else
            promise->reject(ExceptionCode::NotAllowedError);
        return;
    }

    String resultAsString;

    if (type == "text/uri-list"_s) {
        String title;
        resultAsString = activePasteboard().readURL(itemIndex, title).string();
    }

    if (type == textPlainContentTypeAtom()) {
        PasteboardPlainText plainTextReader;
        activePasteboard().read(plainTextReader, PlainTextURLReadingPolicy::IgnoreURL, itemIndex);
        resultAsString = WTFMove(plainTextReader.text);
    }

    if (type == textHTMLContentTypeAtom()) {
        WebContentMarkupReader markupReader { *frame };
        activePasteboard().read(markupReader, WebContentReadingPolicy::OnlyRichTextTypes, itemIndex);
        resultAsString = markupReader.takeMarkup();
    }

    // FIXME: Support reading custom data.
    if (updateSessionValidity() == SessionIsValid::No || resultAsString.isNull()) {
        promise->reject(ExceptionCode::NotAllowedError);
        return;
    }

    if (RefPtr page = frame->page())
        resultAsString = page->applyLinkDecorationFiltering(resultAsString, LinkDecorationFilteringTrigger::Paste);

    promise->resolve<IDLInterface<Blob>>(ClipboardItem::blobFromString(frame->document(), resultAsString, type));
}

Clipboard::SessionIsValid Clipboard::updateSessionValidity()
{
    if (!m_activeSession)
        return SessionIsValid::No;

    if (m_activeSession->changeCount != activePasteboard().changeCount()) {
        m_activeSession = std::nullopt;
        return SessionIsValid::No;
    }

    return SessionIsValid::Yes;
}

void Clipboard::write(const Vector<Ref<ClipboardItem>>& items, Ref<DeferredPromise>&& promise)
{
    RefPtr frame = this->frame();
    if (!frame || !shouldProceedWithClipboardWrite(*frame)) {
        promise->reject(ExceptionCode::NotAllowedError);
        return;
    }

    if (auto existingWriter = std::exchange(m_activeItemWriter, ItemWriter::create(*this, WTFMove(promise))))
        existingWriter->invalidate();

    m_activeItemWriter->write(items);
}

void Clipboard::didResolveOrReject(Clipboard::ItemWriter& writer)
{
    if (m_activeItemWriter == &writer)
        m_activeItemWriter = nullptr;
}

LocalFrame* Clipboard::frame() const
{
    return m_navigator ? m_navigator->frame() : nullptr;
}

Pasteboard& Clipboard::activePasteboard()
{
    ASSERT(m_activeSession);
    ASSERT(m_activeSession->pasteboard);
    return *m_activeSession->pasteboard;
}

Clipboard::ItemWriter::ItemWriter(Clipboard& clipboard, Ref<DeferredPromise>&& promise)
    : m_clipboard(clipboard)
    , m_promise(WTFMove(promise))
    , m_pasteboard(Pasteboard::createForCopyAndPaste(PagePasteboardContext::create(clipboard.frame()->pageID())))
{
}

Clipboard::ItemWriter::~ItemWriter() = default;

void Clipboard::ItemWriter::write(const Vector<Ref<ClipboardItem>>& items)
{
    ASSERT(m_promise);
    ASSERT(m_clipboard);
    m_changeCountAtStart = m_pasteboard->changeCount();
    m_dataToWrite.fill(std::nullopt, items.size());
    m_pendingItemCount = items.size();
    for (size_t index = 0; index < items.size(); ++index) {
        items[index]->collectDataForWriting(*m_clipboard, [this, protectedThis = Ref { *this }, index] (auto data) {
            protectedThis->setData(WTFMove(data), index);
            if (!--m_pendingItemCount)
                didSetAllData();
        });
    }
    if (items.isEmpty())
        didSetAllData();
}

void Clipboard::ItemWriter::invalidate()
{
    if (m_promise)
        reject();
}

void Clipboard::ItemWriter::setData(std::optional<PasteboardCustomData>&& data, size_t index)
{
    if (index >= m_dataToWrite.size()) {
        ASSERT_NOT_REACHED();
        return;
    }

    m_dataToWrite[index] = WTFMove(data);
}

void Clipboard::ItemWriter::didSetAllData()
{
    if (!m_promise)
        return;

    auto newChangeCount = m_pasteboard->changeCount();
    if (m_changeCountAtStart != newChangeCount) {
        // FIXME: Instead of checking the changeCount here, send it over to the client (e.g. the UI process
        // in WebKit2) and perform it there.
        reject();
        return;
    }

    auto dataToWrite = std::exchange(m_dataToWrite, { });
    Vector<PasteboardCustomData> customData;
    customData.reserveInitialCapacity(dataToWrite.size());
    for (auto data : dataToWrite) {
        if (!data) {
            reject();
            return;
        }
        customData.append(*data);
    }

    m_pasteboard->writeCustomData(WTFMove(customData));
    m_promise->resolve();
    m_promise = nullptr;

    if (auto clipboard = std::exchange(m_clipboard, nullptr))
        clipboard->didResolveOrReject(*this);
}

void Clipboard::ItemWriter::reject()
{
    if (auto promise = std::exchange(m_promise, nullptr))
        promise->reject(ExceptionCode::NotAllowedError);

    if (auto clipboard = std::exchange(m_clipboard, nullptr))
        clipboard->didResolveOrReject(*this);
}

}
