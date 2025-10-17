/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 18, 2024.
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

const static uint8_t JSRegExpStringIteratorNumberOfInternalFields = 5;

class JSRegExpStringIterator final : public JSInternalFieldObjectImpl<JSRegExpStringIteratorNumberOfInternalFields> {
public:
    using Base = JSInternalFieldObjectImpl<JSRegExpStringIteratorNumberOfInternalFields>;

    DECLARE_EXPORT_INFO;

    enum class Field : uint8_t {
        RegExp = 0,
        String,
        Global,
        FullUnicode,
        Done,
    };
    static_assert(numberOfInternalFields == JSRegExpStringIteratorNumberOfInternalFields);

    static std::array<JSValue, numberOfInternalFields> initialValues()
    {
        return { {
            jsNull(),
            jsNull(),
            jsBoolean(false),
            jsBoolean(false),
            jsBoolean(false),
        } };
    }

    const WriteBarrier<Unknown>& internalField(Field field) const { return Base::internalField(static_cast<uint32_t>(field)); }
    WriteBarrier<Unknown>& internalField(Field field) { return Base::internalField(static_cast<uint32_t>(field)); }

    template<typename CellType, SubspaceAccess mode>
    static GCClient::IsoSubspace* subspaceFor(VM& vm)
    {
        return vm.regExpStringIteratorSpace<mode>();
    }

    inline static Structure* createStructure(VM&, JSGlobalObject*, JSValue);

    static JSRegExpStringIterator* createWithInitialValues(VM&, Structure*);

    void setRegExp(VM& vm, JSObject* regExp) { internalField(Field::RegExp).set(vm, this, regExp); }
    void setString(VM& vm, JSValue string) { internalField(Field::String).set(vm, this, string); }
    void setGlobal(VM& vm, JSValue global) { internalField(Field::Global).set(vm, this, global); }
    void setFullUnicode(VM& vm, JSValue fullUnicode) { internalField(Field::FullUnicode).set(vm, this, fullUnicode); }

private:
    JSRegExpStringIterator(VM& vm, Structure* structure)
        : Base(vm, structure)
    {
    }

    void finishCreation(VM&);
    DECLARE_VISIT_CHILDREN;

};

STATIC_ASSERT_IS_TRIVIALLY_DESTRUCTIBLE(JSRegExpStringIterator);

JSC_DECLARE_HOST_FUNCTION(regExpStringIteratorPrivateFuncCreate);

} // namespace JSC
