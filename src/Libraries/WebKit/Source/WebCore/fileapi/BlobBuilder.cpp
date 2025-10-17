/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 21, 2022.
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
#include "BlobBuilder.h"
#include "EndingType.h"

#include "Blob.h"
#include <JavaScriptCore/ArrayBuffer.h>
#include <JavaScriptCore/ArrayBufferView.h>
#include <pal/text/TextCodecUTF8.h>
#include <wtf/text/CString.h>
#include <wtf/text/LineEnding.h>

namespace WebCore {

BlobBuilder::BlobBuilder(EndingType endings)
    : m_endings(endings)
{
}

void BlobBuilder::append(RefPtr<ArrayBuffer>&& arrayBuffer)
{
    if (!arrayBuffer)
        return;
    m_appendableData.append(arrayBuffer->span());
}

void BlobBuilder::append(RefPtr<ArrayBufferView>&& arrayBufferView)
{
    if (!arrayBufferView)
        return;
    m_appendableData.append(arrayBufferView->span());
}

void BlobBuilder::append(RefPtr<Blob>&& blob)
{
    if (!blob)
        return;
    if (!m_appendableData.isEmpty())
        m_items.append(BlobPart(WTFMove(m_appendableData)));
    m_items.append(BlobPart(blob->url()));
}

void BlobBuilder::append(const String& text)
{
    auto bytes = PAL::TextCodecUTF8::encodeUTF8(text);

    if (m_endings == EndingType::Native)
        bytes = normalizeLineEndingsToNative(WTFMove(bytes));

    if (m_appendableData.isEmpty())
        m_appendableData = WTFMove(bytes);
    else {
        // FIXME: Would it be better to move multiple vectors into m_items instead of merging them into one?
        m_appendableData.appendVector(bytes);
    }
}

Vector<BlobPart> BlobBuilder::finalize()
{
    if (!m_appendableData.isEmpty())
        m_items.append(BlobPart(WTFMove(m_appendableData)));
    return WTFMove(m_items);
}

} // namespace WebCore
