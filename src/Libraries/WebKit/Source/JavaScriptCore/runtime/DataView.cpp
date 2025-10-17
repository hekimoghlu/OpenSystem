/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 25, 2023.
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
#include "DataView.h"

#include "HeapInlines.h"
#include "JSDataView.h"
#include "JSGlobalObjectInlines.h"

namespace JSC {

DataView::DataView(RefPtr<ArrayBuffer>&& buffer, size_t byteOffset, std::optional<size_t> byteLength)
    : ArrayBufferView(TypeDataView, WTFMove(buffer), byteOffset, byteLength)
{
}

Ref<DataView> DataView::create(RefPtr<ArrayBuffer>&& buffer, size_t byteOffset, std::optional<size_t> byteLength)
{
    return adoptRef(*new DataView(WTFMove(buffer), byteOffset, byteLength));
}

Ref<DataView> DataView::create(RefPtr<ArrayBuffer>&& buffer)
{
    size_t byteLength = buffer->byteLength();
    return create(WTFMove(buffer), 0, byteLength);
}

RefPtr<DataView> DataView::wrappedAs(Ref<ArrayBuffer>&& buffer, size_t byteOffset, std::optional<size_t> byteLength)
{
    ASSERT(byteLength || buffer->isResizableOrGrowableShared());

    // We do not check verifySubRangeLength for resizable buffer case since this function is only called from already created JS DataViews.
    // It is possible that verifySubRangeLength fails when underlying ArrayBuffer is resized, but it is OK since it will be just recognized as OOB DataView.
    if (!buffer->isResizableOrGrowableShared()) {
        if (!ArrayBufferView::verifySubRangeLength(buffer->byteLength(), byteOffset, byteLength.value_or(0), 1))
            return nullptr;
    } else if (buffer->isGrowableShared()) {
        // For growable buffer, we extra-check whether byteOffset and length are within maxByteLength.
        // This does not hit in normal condition, just extra hardening.
        if (!ArrayBufferView::verifySubRangeLength(buffer->maxByteLength().value(), byteOffset, byteLength.value_or(0), 1))
            return nullptr;
    }

    return adoptRef(*new DataView(WTFMove(buffer), byteOffset, byteLength));
}

JSArrayBufferView* DataView::wrapImpl(JSGlobalObject* lexicalGlobalObject, JSGlobalObject* globalObject)
{
    return JSDataView::create(
        lexicalGlobalObject, globalObject->typedArrayStructure(TypeDataView, isResizableOrGrowableShared()), possiblySharedBuffer(), byteOffsetRaw(),
        isAutoLength() ? std::nullopt : std::optional { byteLengthRaw() });
}

} // namespace JSC

