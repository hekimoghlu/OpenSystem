/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 19, 2024.
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
#include "ClipboardItemBindingsDataSource.h"

#include "BitmapImage.h"
#include "Blob.h"
#include "Clipboard.h"
#include "ClipboardItem.h"
#include "CommonAtomStrings.h"
#include "Document.h"
#include "ExceptionCode.h"
#include "FileReaderLoader.h"
#include "GraphicsContext.h"
#include "ImageBuffer.h"
#include "JSBlob.h"
#include "JSDOMPromise.h"
#include "JSDOMPromiseDeferred.h"
#include "LocalFrame.h"
#include "Page.h"
#include "PasteboardCustomData.h"
#include "SharedBuffer.h"
#include "markup.h"
#include <wtf/Function.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

static RefPtr<Document> documentFromClipboard(const Clipboard* clipboard)
{
    if (!clipboard)
        return nullptr;

    auto* frame = clipboard->frame();
    if (!frame)
        return nullptr;

    return frame->document();
}

static FileReaderLoader::ReadType readTypeForMIMEType(const String& type)
{
    if (type == "text/uri-list"_s || type == textPlainContentTypeAtom() || type == textHTMLContentTypeAtom())
        return FileReaderLoader::ReadAsText;
    return FileReaderLoader::ReadAsArrayBuffer;
}

WTF_MAKE_TZONE_ALLOCATED_IMPL(ClipboardItemBindingsDataSource);

ClipboardItemBindingsDataSource::ClipboardItemBindingsDataSource(ClipboardItem& item, Vector<KeyValuePair<String, Ref<DOMPromise>>>&& itemPromises)
    : ClipboardItemDataSource(item)
    , m_itemPromises(WTFMove(itemPromises))
{
}

ClipboardItemBindingsDataSource::~ClipboardItemBindingsDataSource() = default;

Vector<String> ClipboardItemBindingsDataSource::types() const
{
    return m_itemPromises.map([&](auto& typeAndItem) {
        return typeAndItem.key;
    });
}

void ClipboardItemBindingsDataSource::getType(const String& type, Ref<DeferredPromise>&& promise)
{
    auto matchIndex = m_itemPromises.findIf([&](auto& item) {
        return type == item.key;
    });

    if (matchIndex == notFound) {
        promise->reject(ExceptionCode::NotFoundError);
        return;
    }

    Ref itemPromise = m_itemPromises[matchIndex].value;
    itemPromise->whenSettled([itemPromise, promise = WTFMove(promise), type] () mutable {
        if (itemPromise->status() != DOMPromise::Status::Fulfilled) {
            promise->reject(ExceptionCode::AbortError);
            return;
        }

        auto result = itemPromise->result();
        if (!result) {
            promise->reject(ExceptionCode::TypeError);
            return;
        }

        String string;
        result.getString(itemPromise->globalObject(), string);
        if (!string.isNull()) {
            promise->resolve<IDLInterface<Blob>>(ClipboardItem::blobFromString(promise->scriptExecutionContext(), string, type));
            return;
        }

        if (!result.isObject()) {
            promise->reject(ExceptionCode::TypeError);
            return;
        }

        if (auto blob = JSBlob::toWrapped(result.getObject()->vm(), result.getObject()))
            promise->resolve<IDLInterface<Blob>>(*blob);
        else
            promise->reject(ExceptionCode::TypeError);
    });
}

void ClipboardItemBindingsDataSource::clearItemTypeLoaders()
{
    for (auto& itemTypeLoader : m_itemTypeLoaders)
        itemTypeLoader->invokeCompletionHandler();

    m_itemTypeLoaders.clear();
}

void ClipboardItemBindingsDataSource::collectDataForWriting(Clipboard& destination, CompletionHandler<void(std::optional<PasteboardCustomData>)>&& completion)
{
    clearItemTypeLoaders();
    ASSERT(!m_completionHandler);
    m_completionHandler = WTFMove(completion);
    m_writingDestination = destination;
    m_numberOfPendingClipboardTypes = m_itemPromises.size();
    m_itemTypeLoaders = m_itemPromises.map([&](auto& typeAndItem) {
        auto type = typeAndItem.key;
        auto itemTypeLoader = ClipboardItemTypeLoader::create(destination, type, [this, protectedItem = Ref { m_item.get() }] {
            ASSERT(m_numberOfPendingClipboardTypes);
            if (!--m_numberOfPendingClipboardTypes)
                invokeCompletionHandler();
        });

        auto promise = typeAndItem.value;
        /* hack: gcc 8.4 will segfault if the WeakPtr is instantiated within the lambda captures */
        auto wl = WeakPtr { itemTypeLoader };
        promise->whenSettled([this, protectedItem = Ref { m_item.get() }, destination = m_writingDestination, promise, type, weakItemTypeLoader = WTFMove(wl)] () mutable {
            if (!weakItemTypeLoader)
                return;

            Ref itemTypeLoader { *weakItemTypeLoader };
            ASSERT_UNUSED(this, notFound != m_itemTypeLoaders.findIf([&] (auto& loader) { return loader.ptr() == itemTypeLoader.ptr(); }));

            auto result = promise->result();
            if (!result) {
                itemTypeLoader->didFailToResolve();
                return;
            }

            RefPtr clipboard = destination.get();
            if (!clipboard) {
                itemTypeLoader->didFailToResolve();
                return;
            }

            if (!clipboard->scriptExecutionContext()) {
                itemTypeLoader->didFailToResolve();
                return;
            }

            String text;
            result.getString(promise->globalObject(), text);
            if (!text.isNull()) {
                itemTypeLoader->didResolveToString(text);
                return;
            }

            if (!result.isObject()) {
                itemTypeLoader->didFailToResolve();
                return;
            }

            if (RefPtr blob = JSBlob::toWrapped(result.getObject()->vm(), result.getObject()))
                itemTypeLoader->didResolveToBlob(*clipboard->scriptExecutionContext(), blob.releaseNonNull());
            else
                itemTypeLoader->didFailToResolve();
        });

        return itemTypeLoader;
    });

    if (!m_numberOfPendingClipboardTypes)
        invokeCompletionHandler();
}

void ClipboardItemBindingsDataSource::invokeCompletionHandler()
{
    if (!m_completionHandler) {
        ASSERT_NOT_REACHED();
        return;
    }

    auto completionHandler = std::exchange(m_completionHandler, { });
    auto itemTypeLoaders = std::exchange(m_itemTypeLoaders, { });
    RefPtr clipboard = m_writingDestination.get();
    m_writingDestination = nullptr;

    auto document = documentFromClipboard(clipboard.get());
    if (!document) {
        completionHandler(std::nullopt);
        return;
    }

    PasteboardCustomData customData;
    for (auto& itemTypeLoader : itemTypeLoaders) {
        auto type = itemTypeLoader->type();
        auto& data = itemTypeLoader->data();
        if (std::holds_alternative<String>(data) && !!std::get<String>(data))
            customData.writeString(type, std::get<String>(data));
        else if (std::holds_alternative<Ref<SharedBuffer>>(data))
            customData.writeData(type, std::get<Ref<SharedBuffer>>(data).copyRef());
        else {
            completionHandler(std::nullopt);
            return;
        }
    }

    customData.setOrigin(document->originIdentifierForPasteboard());
    completionHandler(WTFMove(customData));
}

ClipboardItemBindingsDataSource::ClipboardItemTypeLoader::ClipboardItemTypeLoader(Clipboard& writingDestination, const String& type, CompletionHandler<void()>&& completionHandler)
    : m_type(type)
    , m_completionHandler(WTFMove(completionHandler))
    , m_writingDestination(writingDestination)
{
}

ClipboardItemBindingsDataSource::ClipboardItemTypeLoader::~ClipboardItemTypeLoader()
{
    if (m_blobLoader)
        m_blobLoader->cancel();
    
    ASSERT(!m_completionHandler);
}

void ClipboardItemBindingsDataSource::ClipboardItemTypeLoader::didFinishLoading()
{
    ASSERT(m_blobLoader);
    auto stringResult = readTypeForMIMEType(m_type) == FileReaderLoader::ReadAsText ? m_blobLoader->stringResult() : nullString();
    if (!stringResult.isNull())
        m_data = { stringResult };
    else if (auto arrayBuffer = m_blobLoader->arrayBufferResult())
        m_data = { SharedBuffer::create(arrayBuffer->span()) };
    m_blobLoader = nullptr;
    invokeCompletionHandler();
}

void ClipboardItemBindingsDataSource::ClipboardItemTypeLoader::didFail(ExceptionCode)
{
    ASSERT(m_blobLoader);
    m_blobLoader = nullptr;
    invokeCompletionHandler();
}

String ClipboardItemBindingsDataSource::ClipboardItemTypeLoader::dataAsString() const
{
    if (std::holds_alternative<Ref<SharedBuffer>>(m_data)) {
        auto& buffer = std::get<Ref<SharedBuffer>>(m_data);
        return String::fromUTF8(buffer->span());
    }

    if (std::holds_alternative<String>(m_data))
        return std::get<String>(m_data);

    return { };
}

void ClipboardItemBindingsDataSource::ClipboardItemTypeLoader::sanitizeDataIfNeeded()
{
    if (m_type == textPlainContentTypeAtom() || m_type == "text/uri-list"_s) {
        RefPtr document = documentFromClipboard(m_writingDestination.get());
        if (!document)
            return;

        auto* page = document->page();
        if (!page)
            return;

        auto urlStringToSanitize = dataAsString();
        if (urlStringToSanitize.isEmpty())
            return;

        m_data = { page->applyLinkDecorationFiltering(urlStringToSanitize, LinkDecorationFilteringTrigger::Copy) };
    }

    if (m_type == textHTMLContentTypeAtom()) {
        auto markupToSanitize = dataAsString();
        if (markupToSanitize.isEmpty())
            return;

        m_data = { sanitizeMarkup(markupToSanitize) };
    }

    if (m_type == "image/png"_s) {
        RefPtr<SharedBuffer> bufferToSanitize;
        if (std::holds_alternative<Ref<SharedBuffer>>(m_data))
            bufferToSanitize = std::get<Ref<SharedBuffer>>(m_data).ptr();
        else if (std::holds_alternative<String>(m_data))
            bufferToSanitize = utf8Buffer(std::get<String>(m_data));

        if (!bufferToSanitize || bufferToSanitize->isEmpty())
            return;

        auto bitmapImage = BitmapImage::create();
        bitmapImage->setData(WTFMove(bufferToSanitize), true);
        auto imageBuffer = ImageBuffer::create(bitmapImage->size(), RenderingMode::Unaccelerated, RenderingPurpose::Unspecified, 1, DestinationColorSpace::SRGB(), ImageBufferPixelFormat::BGRA8);
        if (!imageBuffer) {
            m_data = { nullString() };
            return;
        }

        imageBuffer->context().drawImage(bitmapImage.get(), FloatPoint::zero());
        m_data = { SharedBuffer::create(imageBuffer->toData("image/png"_s)) };
    }
}

void ClipboardItemBindingsDataSource::ClipboardItemTypeLoader::invokeCompletionHandler()
{
    if (auto completion = WTFMove(m_completionHandler)) {
        sanitizeDataIfNeeded();
        completion();
    }
}

void ClipboardItemBindingsDataSource::ClipboardItemTypeLoader::didResolveToBlob(ScriptExecutionContext& context, Ref<Blob>&& blob)
{
    ASSERT(!m_blobLoader);
    m_blobLoader = makeUnique<FileReaderLoader>(readTypeForMIMEType(m_type), this);
    m_blobLoader->start(&context, WTFMove(blob));
}

void ClipboardItemBindingsDataSource::ClipboardItemTypeLoader::didFailToResolve()
{
    ASSERT(!m_blobLoader);
    invokeCompletionHandler();
}

void ClipboardItemBindingsDataSource::ClipboardItemTypeLoader::didResolveToString(const String& text)
{
    ASSERT(!m_blobLoader);
    m_data = { text };
    invokeCompletionHandler();
}

} // namespace WebCore
