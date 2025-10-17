/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 12, 2024.
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

#include "JSInternalFieldObjectImpl.h"

namespace JSC {

// This class is used as a base for classes such as String,
// Number, Boolean and Symbol which are wrappers for primitive types.
class JSWrapperObject : public JSInternalFieldObjectImpl<1> {
public:
    using Base = JSInternalFieldObjectImpl<1>;

    template<typename, SubspaceAccess>
    static void subspaceFor(VM&)
    {
        RELEASE_ASSERT_NOT_REACHED();
    }

    static size_t allocationSize(Checked<size_t> inlineCapacity)
    {
        ASSERT_UNUSED(inlineCapacity, !inlineCapacity);
        return sizeof(JSWrapperObject);
    }

    enum class Field : uint32_t {
        WrappedValue = 0,
    };
    static_assert(numberOfInternalFields == 1);

    JSValue internalValue() const;
    void setInternalValue(VM&, JSValue);

    static constexpr ptrdiff_t internalValueOffset() { return offsetOfInternalField(static_cast<unsigned>(Field::WrappedValue)); }
    static constexpr ptrdiff_t internalValueCellOffset()
    {
#if USE(JSVALUE64)
        return internalValueOffset();
#else
        return internalValueOffset() + OBJECT_OFFSETOF(EncodedValueDescriptor, asBits.payload);
#endif
    }

protected:
    explicit JSWrapperObject(VM&, Structure*);

    DECLARE_VISIT_CHILDREN_WITH_MODIFIER(JS_EXPORT_PRIVATE);
};

inline JSWrapperObject::JSWrapperObject(VM& vm, Structure* structure)
    : Base(vm, structure)
{
}

inline JSValue JSWrapperObject::internalValue() const
{
    return internalField(static_cast<unsigned>(Field::WrappedValue)).get();
}

inline void JSWrapperObject::setInternalValue(VM& vm, JSValue value)
{
    ASSERT(value);
    ASSERT(!value.isObject());
    internalField(static_cast<unsigned>(Field::WrappedValue)).set(vm, this, value);
}

} // namespace JSC
