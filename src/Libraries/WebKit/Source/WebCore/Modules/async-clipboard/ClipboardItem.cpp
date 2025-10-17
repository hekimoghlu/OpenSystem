/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 8, 2022.
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
#include "ClipboardItem.h"

#include "Blob.h"
#include "Clipboard.h"
#include "ClipboardItemBindingsDataSource.h"
#include "ClipboardItemPasteboardDataSource.h"
#include "CommonAtomStrings.h"
#include "Navigator.h"
#include "PasteboardCustomData.h"
#include "PasteboardItemInfo.h"
#include "SharedBuffer.h"

namespace WebCore {

ClipboardItem::~ClipboardItem() = default;

Ref<Blob> ClipboardItem::blobFromString(ScriptExecutionContext* context, const String& stringData, const String& type)
{
    return Blob::create(context, Vector(stringData.utf8().span()), Blob::normalizedContentType(type));
}

static ClipboardItem::PresentationStyle clipboardItemPresentationStyle(const PasteboardItemInfo& info)
{
    switch (info.preferredPresentationStyle) {
    case PasteboardItemPresentationStyle::Unspecified:
        return ClipboardItem::PresentationStyle::Unspecified;
    case PasteboardItemPresentationStyle::Inline:
        return ClipboardItem::PresentationStyle::Inline;
    case PasteboardItemPresentationStyle::Attachment:
        return ClipboardItem::PresentationStyle::Attachment;
    }
    ASSERT_NOT_REACHED();
    return ClipboardItem::PresentationStyle::Unspecified;
}

// FIXME: Custom format starts with `"web "`("web" followed by U+0020 SPACE) prefix
// and suffix (after stripping out `"web "`) passes the parsing a MIME type check.
// https://w3c.github.io/clipboard-apis/#optional-data-types
// https://webkit.org/b/280664
ClipboardItem::ClipboardItem(Vector<KeyValuePair<String, Ref<DOMPromise>>>&& items, const Options& options)
    : m_dataSource(makeUnique<ClipboardItemBindingsDataSource>(*this, WTFMove(items)))
    , m_presentationStyle(options.presentationStyle)
{
}

ClipboardItem::ClipboardItem(Clipboard& clipboard, const PasteboardItemInfo& info)
    : m_clipboard(clipboard)
    , m_navigator(clipboard.navigator())
    , m_dataSource(makeUnique<ClipboardItemPasteboardDataSource>(*this, info))
    , m_presentationStyle(clipboardItemPresentationStyle(info))
{
}

ExceptionOr<Ref<ClipboardItem>> ClipboardItem::create(Vector<KeyValuePair<String, Ref<DOMPromise>>>&& data, const Options& options)
{
    if (data.isEmpty())
        return Exception { ExceptionCode::TypeError, "ClipboardItem() can not be an empty array: {}"_s };
    return adoptRef(*new ClipboardItem(WTFMove(data), options));
}

Ref<ClipboardItem> ClipboardItem::create(Clipboard& clipboard, const PasteboardItemInfo& info)
{
    return adoptRef(*new ClipboardItem(clipboard, info));
}

Vector<String> ClipboardItem::types() const
{
    return m_dataSource->types();
}

void ClipboardItem::getType(const String& type, Ref<DeferredPromise>&& promise)
{
    m_dataSource->getType(type, WTFMove(promise));
}

bool ClipboardItem::supports(const String& type)
{
    // FIXME: Custom format starts with `"web "`("web" followed by U+0020 SPACE) prefix
    // and suffix (after stripping out `"web "`) passes the parsing a MIME type check.
    // https://webkit.org/b/280664
    // FIXME: add type == "image/svg+xml"_s when we have sanitized copy/paste for SVG data
    // https://webkit.org/b/280726
    if (type == textPlainContentTypeAtom()
        || type == textHTMLContentTypeAtom()
        || type == "image/png"_s
        || type == "text/uri-list"_s) {
        return true;
        }
    return false;
}

void ClipboardItem::collectDataForWriting(Clipboard& destination, CompletionHandler<void(std::optional<PasteboardCustomData>)>&& completion)
{
    m_dataSource->collectDataForWriting(destination, WTFMove(completion));
}

Navigator* ClipboardItem::navigator()
{
    return m_navigator.get();
}

Clipboard* ClipboardItem::clipboard()
{
    return m_clipboard.get();
}

} // namespace WebCore
