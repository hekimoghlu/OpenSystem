/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 8, 2024.
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

#include "ArrayBuffer.h"
#include "ArrayBufferView.h"

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace JSC {

template<typename Adaptor>
class GenericTypedArrayView final : public ArrayBufferView {
public:
    static Ref<GenericTypedArrayView> create(size_t length);
    static Ref<GenericTypedArrayView> create(const typename Adaptor::Type* array, size_t length);
    static Ref<GenericTypedArrayView> create(std::span<const typename Adaptor::Type> data) { return create(data.data(), data.size()); }
    static Ref<GenericTypedArrayView> create(Ref<ArrayBuffer>&&);
    static Ref<GenericTypedArrayView> create(RefPtr<ArrayBuffer>&&, size_t byteOffset, std::optional<size_t> length);
    static RefPtr<GenericTypedArrayView> tryCreate(size_t length);
    static RefPtr<GenericTypedArrayView> tryCreate(const typename Adaptor::Type* array, size_t length);
    static RefPtr<GenericTypedArrayView> tryCreate(std::span<const typename Adaptor::Type> data) { return tryCreate(data.data(), data.size()); }
    static RefPtr<GenericTypedArrayView> tryCreate(RefPtr<ArrayBuffer>&&, size_t byteOffset, std::optional<size_t> length);
    
    static Ref<GenericTypedArrayView> createUninitialized(size_t length);
    static RefPtr<GenericTypedArrayView> tryCreateUninitialized(size_t length);
    
    typename Adaptor::Type* data() const { return static_cast<typename Adaptor::Type*>(baseAddress()); }

    std::span<const typename Adaptor::Type> typedSpan() const { return unsafeMakeSpan(data(), length()); }
    std::span<typename Adaptor::Type> typedMutableSpan() { return unsafeMakeSpan(data(), length()); }

    bool set(GenericTypedArrayView<Adaptor>* array, size_t offset)
    {
        return setImpl(array, offset * sizeof(typename Adaptor::Type));
    }
    
    bool setRange(const typename Adaptor::Type* data, size_t count, size_t offset)
    {
        return setRangeImpl(
            reinterpret_cast<const char*>(data),
            count * sizeof(typename Adaptor::Type),
            offset * sizeof(typename Adaptor::Type));
    }
    
    bool zeroRange(size_t offset, size_t count)
    {
        return zeroRangeImpl(offset * sizeof(typename Adaptor::Type), count * sizeof(typename Adaptor::Type));
    }
    
    void zeroFill() { zeroRange(0, length()); }
    
    size_t length() const
    {
        return byteLength() / sizeof(typename Adaptor::Type);
    }

    size_t lengthRaw() const
    {
        return byteLengthRaw() / sizeof(typename Adaptor::Type);
    }

    typename Adaptor::Type item(size_t index) const
    {
        ASSERT_WITH_SECURITY_IMPLICATION(index < this->length());
        return data()[index];
    }
    
    void set(size_t index, double value) const
    {
        ASSERT_WITH_SECURITY_IMPLICATION(index < this->length());
        data()[index] = Adaptor::toNativeFromDouble(value);
    }

    void setNative(size_t index, typename Adaptor::Type value) const
    {
        ASSERT_WITH_SECURITY_IMPLICATION(index < this->length());
        data()[index] = value;
    }

    bool getRange(typename Adaptor::Type* data, size_t count, size_t offset)
    {
        return getRangeImpl(
            reinterpret_cast<char*>(data),
            count * sizeof(typename Adaptor::Type),
            offset * sizeof(typename Adaptor::Type));
    }

    bool checkInboundData(size_t offset, size_t count) const
    {
        return isSumSmallerThanOrEqual(offset, count, this->length());
    }

    JSArrayBufferView* wrapImpl(JSGlobalObject* lexicalGlobalObject, JSGlobalObject* globalObject);

    static RefPtr<GenericTypedArrayView<Adaptor>> wrappedAs(Ref<ArrayBuffer>&&, size_t byteOffset, std::optional<size_t> length);

private:
    GenericTypedArrayView(RefPtr<ArrayBuffer>&&, size_t byteOffset, std::optional<size_t> length);
};

} // namespace JSC

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
