/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 5, 2025.
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
#include "JSDataView.h"

#include "JSCInlines.h"
#include "TypeError.h"

namespace JSC {

const ClassInfo JSDataView::s_info = { "DataView"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(JSDataView) };
const ClassInfo JSResizableOrGrowableSharedDataView::s_info = { "DataView"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(JSResizableOrGrowableSharedDataView) };

JSDataView::JSDataView(VM& vm, ConstructionContext& context, ArrayBuffer* buffer)
    : Base(vm, context)
    , m_buffer(buffer)
{
}

JSDataView* JSDataView::create(
    JSGlobalObject* globalObject, Structure* structure, RefPtr<ArrayBuffer>&& buffer,
    size_t byteOffset, std::optional<size_t> byteLength)
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    ASSERT(buffer);
    if (buffer->isDetached()) {
        throwTypeError(globalObject, scope, typedArrayBufferHasBeenDetachedErrorMessage);
        return nullptr;
    }

    ASSERT(byteLength || buffer->isResizableOrGrowableShared());

    if (!ArrayBufferView::verifySubRangeLength(buffer->byteLength(), byteOffset, byteLength.value_or(0), sizeof(uint8_t))) {
        throwRangeError(globalObject, scope, "Length out of range of buffer"_s);
        return nullptr;
    }

    if (!ArrayBufferView::verifyByteOffsetAlignment(byteOffset, sizeof(uint8_t))) {
        throwRangeError(globalObject, scope, "Byte offset is not aligned"_s);
        return nullptr;
    }

    ConstructionContext context(
        structure, buffer.copyRef(), byteOffset, byteLength, ConstructionContext::DataView);
    ASSERT(context);
    JSDataView* result =
        new (NotNull, allocateCell<JSDataView>(vm)) JSDataView(vm, context, buffer.get());
    result->finishCreation(vm);
    return result;
}

JSDataView* JSDataView::createUninitialized(JSGlobalObject*, Structure*, size_t)
{
    UNREACHABLE_FOR_PLATFORM();
    return nullptr;
}

JSDataView* JSDataView::create(JSGlobalObject*, Structure*, size_t)
{
    UNREACHABLE_FOR_PLATFORM();
    return nullptr;
}

bool JSDataView::setFromTypedArray(JSGlobalObject*, size_t, JSArrayBufferView*, size_t, size_t, CopyType)
{
    UNREACHABLE_FOR_PLATFORM();
    return false;
}

bool JSDataView::setFromArrayLike(JSGlobalObject*, size_t, JSObject*, size_t, size_t)
{
    UNREACHABLE_FOR_PLATFORM();
    return false;
}

bool JSDataView::setIndex(JSGlobalObject*, size_t, JSValue)
{
    UNREACHABLE_FOR_PLATFORM();
    return false;
}

RefPtr<DataView> JSDataView::possiblySharedTypedImpl()
{
    return DataView::create(possiblySharedBuffer(), byteOffsetRaw(), isAutoLength() ? std::nullopt : std::optional { lengthRaw() });
}

RefPtr<DataView> JSDataView::unsharedTypedImpl()
{
    return DataView::create(unsharedBuffer(), byteOffsetRaw(), isAutoLength() ? std::nullopt : std::optional { lengthRaw() });
}

Structure* JSDataView::createStructure(VM& vm, JSGlobalObject* globalObject, JSValue prototype)
{
    return Structure::create(vm, globalObject, prototype, TypeInfo(DataViewType, StructureFlags), info(), NonArray);
}

Structure* JSResizableOrGrowableSharedDataView::createStructure(VM& vm, JSGlobalObject* globalObject, JSValue prototype)
{
    return Structure::create(vm, globalObject, prototype, TypeInfo(DataViewType, StructureFlags), info(), NonArray);
}

} // namespace JSC

