/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 11, 2024.
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

#include "JSObject.h"

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace JSC {

// This is used for sharing interface and implementation. It should not have its own classInfo.
template<unsigned passedNumberOfInternalFields = 1>
class JSInternalFieldObjectImpl : public JSNonFinalObject {
public:
    friend class LLIntOffsetsExtractor;

    using Base = JSNonFinalObject;
    static constexpr unsigned numberOfInternalFields = passedNumberOfInternalFields;

    template<typename CellType, SubspaceAccess>
    static void subspaceFor(VM&)
    {
        RELEASE_ASSERT_NOT_REACHED();
    }

    static size_t allocationSize(Checked<size_t> inlineCapacity)
    {
        ASSERT_UNUSED(inlineCapacity, !inlineCapacity);
        return sizeof(JSInternalFieldObjectImpl);
    }

    const WriteBarrier<Unknown>& internalField(unsigned index) const
    {
        ASSERT(index < numberOfInternalFields);
        return m_internalFields[index];
    }

    WriteBarrier<Unknown>& internalField(unsigned index)
    {
        ASSERT(index < numberOfInternalFields);
        return m_internalFields[index];
    }

    static constexpr ptrdiff_t offsetOfInternalFields() { return OBJECT_OFFSETOF(JSInternalFieldObjectImpl, m_internalFields); }
    static constexpr ptrdiff_t offsetOfInternalField(unsigned index) { return OBJECT_OFFSETOF(JSInternalFieldObjectImpl, m_internalFields) + index * sizeof(WriteBarrier<Unknown>); }

protected:
    DECLARE_VISIT_CHILDREN;

    JSInternalFieldObjectImpl(VM& vm, Structure* structure)
        : Base(vm, structure)
    {
    }

    WriteBarrier<Unknown> m_internalFields[numberOfInternalFields] { };
};

} // namespace JSC

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
