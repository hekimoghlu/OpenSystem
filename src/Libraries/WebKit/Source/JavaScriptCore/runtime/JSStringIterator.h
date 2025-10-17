/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 18, 2022.
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

class JSStringIterator final : public JSInternalFieldObjectImpl<2> {
public:
    using Base = JSInternalFieldObjectImpl<2>;

    enum class Field : uint8_t {
        Index = 0,
        IteratedString,
    };

    DECLARE_EXPORT_INFO;

    static size_t allocationSize(Checked<size_t> inlineCapacity)
    {
        ASSERT_UNUSED(inlineCapacity, inlineCapacity == 0U);
        return sizeof(JSStringIterator);
    }

    template<typename CellType, SubspaceAccess mode>
    static GCClient::IsoSubspace* subspaceFor(VM& vm)
    {
        return vm.stringIteratorSpace<mode>();
    }

    static std::array<JSValue, numberOfInternalFields> initialValues()
    {
        return { {
            jsNumber(0),
            jsNull(),
        } };
    }

    const WriteBarrier<Unknown>& internalField(Field field) const { return Base::internalField(static_cast<uint32_t>(field)); }
    WriteBarrier<Unknown>& internalField(Field field) { return Base::internalField(static_cast<uint32_t>(field)); }

    inline static Structure* createStructure(VM&, JSGlobalObject*, JSValue);

    static JSStringIterator* create(VM& vm, Structure* structure, JSString* iteratedString)
    {
        JSStringIterator* instance = new (NotNull, allocateCell<JSStringIterator>(vm)) JSStringIterator(vm, structure);
        instance->finishCreation(vm, iteratedString);
        return instance;
    }

    JSValue iteratedString() const { return internalField(Field::IteratedString).get(); }
    JSValue index() const { return internalField(Field::Index).get(); }
    JSStringIterator* clone(JSGlobalObject*);

    DECLARE_VISIT_CHILDREN;

private:
    JSStringIterator(VM& vm, Structure* structure)
        : Base(vm, structure)
    {
    }

    void finishCreation(VM&, JSString* iteratedString);
};

} // namespace JSC
