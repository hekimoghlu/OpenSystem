/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 21, 2022.
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
#pragma once

#include "DataView.h"
#include "JSArrayBufferView.h"

namespace JSC {

class JSDataView : public JSArrayBufferView {
public:
    using Base = JSArrayBufferView;
    static constexpr unsigned StructureFlags = Base::StructureFlags;

    static constexpr unsigned elementSize = 1;

    static constexpr TypedArrayContentType contentType = TypedArrayContentType::None;

    template<typename CellType, SubspaceAccess mode>
    static GCClient::IsoSubspace* subspaceFor(VM& vm)
    {
        return vm.dataViewSpace<mode>();
    }

    JS_EXPORT_PRIVATE static JSDataView* create(JSGlobalObject*, Structure*, RefPtr<ArrayBuffer>&&, size_t byteOffset, std::optional<size_t> byteLength);
    
    // Dummy methods, which don't actually work; these are just in place to
    // placate some template specialization we do elsewhere.
    static JSDataView* createUninitialized(JSGlobalObject*, Structure*, size_t length);
    static JSDataView* create(JSGlobalObject*, Structure*, size_t length);
    bool setFromTypedArray(JSGlobalObject*, size_t offset, JSArrayBufferView*, size_t objectOffset, size_t length, CopyType);
    bool setFromArrayLike(JSGlobalObject*, size_t offset, JSObject*, size_t objectOffset, size_t length);
    bool setIndex(JSGlobalObject*, size_t, JSValue);

    template<typename Getter>
    std::optional<size_t> viewByteLength(Getter& getter)
    {
        // https://tc39.es/proposal-resizablearraybuffer/#sec-isviewoutofbounds
        // https://tc39.es/proposal-resizablearraybuffer/#sec-getviewbytelength
        if (UNLIKELY(isDetached()))
            return std::nullopt;

        if (LIKELY(canUseRawFieldsDirectly()))
            return byteLengthRaw();

        RefPtr<ArrayBuffer> buffer = possiblySharedBuffer();
        if (!buffer)
            return 0;

        size_t bufferByteLength = getter(*buffer);
        size_t byteOffset = byteOffsetRaw();
        size_t byteLength = byteLengthRaw() + byteOffset; // Keep in mind that byteLengthRaw returns 0 for AutoLength TypedArray.
        if (byteLength > bufferByteLength)
            return std::nullopt;
        if (isAutoLength())
            return bufferByteLength - byteOffset;
        return byteLengthRaw();
    }
    
    ArrayBuffer* possiblySharedBuffer() const { return m_buffer; }
    ArrayBuffer* unsharedBuffer() const
    {
        RELEASE_ASSERT(!m_buffer->isShared());
        return m_buffer;
    }
    static constexpr ptrdiff_t offsetOfBuffer() { return OBJECT_OFFSETOF(JSDataView, m_buffer); }
    
    RefPtr<DataView> possiblySharedTypedImpl();
    RefPtr<DataView> unsharedTypedImpl();
    
    static constexpr TypedArrayType TypedArrayStorageType = TypeDataView;

    static Structure* createStructure(VM&, JSGlobalObject*, JSValue prototype);
    
    DECLARE_EXPORT_INFO;

private:
    JSDataView(VM&, ConstructionContext&, ArrayBuffer*);

    ArrayBuffer* m_buffer;
};

class JSResizableOrGrowableSharedDataView final : public JSDataView {
public:
    using Base = JSDataView;
    using Base::StructureFlags;

    static constexpr bool isResizableOrGrowableSharedTypedArray = true;

    DECLARE_EXPORT_INFO;

    static Structure* createStructure(VM&, JSGlobalObject*, JSValue prototype);
};

} // namespace JSC
