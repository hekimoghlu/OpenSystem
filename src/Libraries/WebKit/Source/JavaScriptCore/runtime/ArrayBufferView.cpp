/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 15, 2021.
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
#include "ArrayBufferView.h"

#include "DataView.h"
#include "TypedArrayInlines.h"

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace JSC {

ArrayBufferView::ArrayBufferView(TypedArrayType type, RefPtr<ArrayBuffer>&& buffer, size_t byteOffset, std::optional<size_t> byteLength)
    : m_type(type)
    , m_isResizableNonShared(buffer->isResizableNonShared())
    , m_isGrowableShared(buffer->isGrowableShared())
    , m_isAutoLength(buffer->isResizableOrGrowableShared() && !byteLength)
    , m_byteOffset(byteOffset)
    , m_byteLength(byteLength.value_or(0))
    , m_buffer(WTFMove(buffer))
{
    if (byteLength) {
        // If it is resizable, then it can be possible that length exceeds byteLength, and this is fine since it just becomes OOB array.
        if (!isResizableOrGrowableShared()) {
            Checked<size_t, CrashOnOverflow> length(byteOffset);
            length += byteLength.value();
            RELEASE_ASSERT_WITH_SECURITY_IMPLICATION(length <= m_buffer->byteLength());
        }
    } else
        ASSERT(isAutoLength());

    if (m_buffer)
        m_baseAddress = BaseAddress(static_cast<char*>(m_buffer->data()) + m_byteOffset);
}

template<typename Visitor> constexpr decltype(auto) ArrayBufferView::visitDerived(Visitor&& visitor)
{
    switch (m_type) {
    case TypedArrayType::NotTypedArray:
    case TypedArrayType::TypeDataView:
        return std::invoke(std::forward<Visitor>(visitor), static_cast<DataView&>(*this));
#define DECLARE_TYPED_ARRAY_TYPE(name) \
    case TypedArrayType::Type##name: \
        return std::invoke(std::forward<Visitor>(visitor), static_cast<name##Array&>(*this));
    FOR_EACH_TYPED_ARRAY_TYPE_EXCLUDING_DATA_VIEW(DECLARE_TYPED_ARRAY_TYPE)
#undef DECLARE_TYPED_ARRAY_TYPE
    }
    RELEASE_ASSERT_NOT_REACHED();
}

template<typename Visitor> constexpr decltype(auto) ArrayBufferView::visitDerived(Visitor&& visitor) const
{
    return const_cast<ArrayBufferView&>(*this).visitDerived([&](auto& value) {
        return std::invoke(std::forward<Visitor>(visitor), std::as_const(value));
    });
}

JSArrayBufferView* ArrayBufferView::wrap(JSGlobalObject* lexicalGlobalObject, JSGlobalObject* globalObject)
{
    return visitDerived([&](auto& derived) { return derived.wrapImpl(lexicalGlobalObject, globalObject); });
}

void ArrayBufferView::operator delete(ArrayBufferView* value, std::destroying_delete_t)
{
    value->visitDerived([](auto& value) {
        using T = std::decay_t<decltype(value)>;
        std::destroy_at(&value);
        T::freeAfterDestruction(&value);
    });
}

void ArrayBufferView::setDetachable(bool flag)
{
    if (flag == m_isDetachable)
        return;
    
    m_isDetachable = flag;
    
    if (!m_buffer)
        return;
    
    if (flag)
        m_buffer->unpin();
    else
        m_buffer->pin();
}

} // namespace JSC

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
