/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 13, 2024.
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

const static uint8_t JSWrapForValidIteratorNumberOfInternalFields = 2;

class JSWrapForValidIterator final : public JSInternalFieldObjectImpl<JSWrapForValidIteratorNumberOfInternalFields> {

public:
    using Base = JSInternalFieldObjectImpl<JSWrapForValidIteratorNumberOfInternalFields>;

    DECLARE_EXPORT_INFO;

    enum class Field : uint8_t {
        IteratedIterator = 0,
        IteratedNextMethod,
    };
    static_assert(numberOfInternalFields == JSWrapForValidIteratorNumberOfInternalFields);

    static std::array<JSValue, numberOfInternalFields> initialValues()
    {
        return { {
            jsNull(),
            jsNull(),
        } };
    }

    const WriteBarrier<Unknown>& internalField(Field field) const { return Base::internalField(static_cast<uint32_t>(field)); }
    WriteBarrier<Unknown>& internalField(Field field) { return Base::internalField(static_cast<uint32_t>(field)); }

    template<typename CellType, SubspaceAccess mode>
    static GCClient::IsoSubspace* subspaceFor(VM& vm)
    {
        return vm.wrapForValidIteratorSpace<mode>();
    }

    inline static Structure* createStructure(VM&, JSGlobalObject*, JSValue);

    static JSWrapForValidIterator* createWithInitialValues(VM&, Structure*);
    static JSWrapForValidIterator* create(VM&, Structure*, JSValue iterator, JSValue nextMethod);

    JSObject* iteratedIterator() const { return jsCast<JSObject*>(internalField(Field::IteratedIterator).get()); }
    JSObject* iteratedNextMethod() const { return jsCast<JSObject*>(internalField(Field::IteratedNextMethod).get()); }

    void setIteratedIterator(VM& vm, JSValue iterator) { internalField(Field::IteratedIterator).set(vm, this, iterator); }
    void setIteratedNextMethod(VM& vm, JSValue nextMethod) { internalField(Field::IteratedNextMethod).set(vm, this, nextMethod); }

private:
    JSWrapForValidIterator(VM& vm, Structure* structure)
        : Base(vm, structure)
    {
    }

    void finishCreation(VM&, JSValue iterator, JSValue nextMethod);
    DECLARE_VISIT_CHILDREN;
};

STATIC_ASSERT_IS_TRIVIALLY_DESTRUCTIBLE(JSWrapForValidIterator);

JSC_DECLARE_HOST_FUNCTION(wrapForValidIteratorPrivateFuncCreate);

} // namespace JSC
