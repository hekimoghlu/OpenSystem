/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 22, 2022.
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
#include "ClipboardItemPasteboardDataSource.h"

#include "Clipboard.h"
#include "ClipboardItem.h"
#include "JSDOMPromiseDeferred.h"
#include "PasteboardCustomData.h"
#include "PasteboardItemInfo.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(ClipboardItemPasteboardDataSource);

ClipboardItemPasteboardDataSource::ClipboardItemPasteboardDataSource(ClipboardItem& item, const PasteboardItemInfo& info)
    : ClipboardItemDataSource(item)
    , m_types(info.webSafeTypesByFidelity)
{
}

ClipboardItemPasteboardDataSource::~ClipboardItemPasteboardDataSource() = default;

Vector<String> ClipboardItemPasteboardDataSource::types() const
{
    return m_types;
}

void ClipboardItemPasteboardDataSource::getType(const String& type, Ref<DeferredPromise>&& promise)
{
    if (RefPtr clipboard = m_item->clipboard())
        clipboard->getType(Ref { m_item.get() }, type, WTFMove(promise));
    else
        promise->reject(ExceptionCode::NotAllowedError);
}

void ClipboardItemPasteboardDataSource::collectDataForWriting(Clipboard&, CompletionHandler<void(std::optional<PasteboardCustomData>)>&& completion)
{
    // FIXME: Not implemented. This is needed to support writing platform-backed ClipboardItems
    // back to the pasteboard using Clipboard.write().
    completion(std::nullopt);
}

} // namespace WebCore
