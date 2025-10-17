/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 15, 2023.
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

class JSArrayIterator final : public JSInternalFieldObjectImpl<3> {
public:
    using Base = JSInternalFieldObjectImpl<3>;

    enum class Field : uint8_t {
        Index = 0,
        IteratedObject,
        Kind,
    };
    static_assert(numberOfInternalFields == 3);

    static size_t allocationSize(Checked<size_t> inlineCapacity)
    {
        ASSERT_UNUSED(inlineCapacity, inlineCapacity == 0U);
        return sizeof(JSArrayIterator);
    }
    
    template<typename CellType, SubspaceAccess mode>
    static GCClient::IsoSubspace* subspaceFor(VM& vm)
    {
        return vm.arrayIteratorSpace<mode>();
    }

    static std::array<JSValue, numberOfInternalFields> initialValues()
    {
        return { {
            jsNumber(0),
            jsNull(),
            jsNumber(0),
        } };
    }

    static constexpr int64_t doneIndex = -1;
    const WriteBarrier<Unknown>& internalField(Field field) const { return Base::internalField(static_cast<uint32_t>(field)); }
    WriteBarrier<Unknown>& internalField(Field field) { return Base::internalField(static_cast<uint32_t>(field)); }

    IterationKind kind() const { return static_cast<IterationKind>(internalField(Field::Kind).get().asUInt32AsAnyInt()); }
    JSObject* iteratedObject() const { return jsCast<JSObject*>(internalField(Field::IteratedObject).get()); }

    JS_EXPORT_PRIVATE static JSArrayIterator* create(VM&, Structure*, JSObject* iteratedObject, JSValue kind);
    static JSArrayIterator* create(VM& vm, Structure* structure, JSObject* iteratedObject, IterationKind kind)
    {
        return create(vm, structure, iteratedObject, jsNumber(static_cast<unsigned>(kind)));
    }
    static JSArrayIterator* createWithInitialValues(VM&, Structure*);
    static Structure* createStructure(VM&, JSGlobalObject*, JSValue);

    DECLARE_INFO;

    DECLARE_VISIT_CHILDREN;

private:
    JSArrayIterator(VM&, Structure*);
    void finishCreation(VM&);
};

} // namespace JSC
