/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 11, 2023.
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

#include "IterationKind.h"
#include "JSInternalFieldObjectImpl.h"
#include "JSSet.h"

namespace JSC {

const static uint8_t JSSetIteratorNumberOFInternalFields = 4;

class JSSetIterator final : public JSInternalFieldObjectImpl<JSSetIteratorNumberOFInternalFields> {
public:
    using Base = JSInternalFieldObjectImpl<JSSetIteratorNumberOFInternalFields>;

    DECLARE_EXPORT_INFO;

    enum class Field : uint8_t {
        Entry = 0,
        IteratedObject,
        Storage,
        Kind,
    };
    static_assert(numberOfInternalFields == JSSetIteratorNumberOFInternalFields);

    static std::array<JSValue, numberOfInternalFields> initialValues()
    {
        return { {
            jsNumber(0),
            jsNull(),
            jsNull(),
            jsNumber(0),
        } };
    }

    const WriteBarrier<Unknown>& internalField(Field field) const { return Base::internalField(static_cast<uint32_t>(field)); }
    WriteBarrier<Unknown>& internalField(Field field) { return Base::internalField(static_cast<uint32_t>(field)); }

    template<typename CellType, SubspaceAccess mode>
    static GCClient::IsoSubspace* subspaceFor(VM& vm)
    {
        return vm.setIteratorSpace<mode>();
    }

    inline static Structure* createStructure(VM&, JSGlobalObject*, JSValue);

    static JSSetIterator* create(JSGlobalObject* globalObject, Structure* structure, JSSet* iteratedObject, IterationKind kind)
    {
        VM& vm = getVM(globalObject);
        JSSetIterator* instance = new (NotNull, allocateCell<JSSetIterator>(vm)) JSSetIterator(vm, structure);
        instance->finishCreation(globalObject, iteratedObject, kind);
        return instance;
    }

    static JSSetIterator* createWithInitialValues(VM&, Structure*);

    ALWAYS_INLINE JSValue nextWithAdvance(VM& vm)
    {
        JSCell* storage = this->storage();
        if (storage == vm.orderedHashTableSentinel())
            return { };

        JSSet::Storage& storageRef = *jsCast<JSSet::Storage*>(storage);
        auto result = JSSet::Helper::transitAndNext(vm, storageRef, entry());
        if (!result.storage) {
            setStorage(vm, vm.orderedHashTableSentinel());
            return { };
        }

        setEntry(vm, result.entry + 1);
        if (result.storage != storage)
            setStorage(vm, result.storage);
        return result.key;
    }

    bool next(JSGlobalObject* globalObject, JSValue& value)
    {
        JSValue nextKey = nextWithAdvance(globalObject->vm());
        if (nextKey.isEmpty())
            return false;

        switch (kind()) {
        case IterationKind::Values:
        case IterationKind::Keys:
            value = nextKey;
            break;
        case IterationKind::Entries:
            value = createTuple(globalObject, nextKey, nextKey);
            break;
        }
        return true;
    }

    JSValue next(VM& vm)
    {
        JSValue key = nextWithAdvance(vm);
        return key.isEmpty() ? jsBoolean(true) : jsBoolean(false);
    }
    JSValue nextKey(VM& vm)
    {
        JSSet::Helper::Entry entry = this->entry() - 1;
        JSCell* storage = this->storage();
        ASSERT_UNUSED(vm, storage != vm.orderedHashTableSentinel());
        JSSet::Storage& storageRef = *jsCast<JSSet::Storage*>(storage);
        return JSSet::Helper::getKey(storageRef, entry);
    }

    IterationKind kind() const { return static_cast<IterationKind>(internalField(Field::Kind).get().asUInt32AsAnyInt()); }
    JSObject* iteratedObject() const { return jsCast<JSObject*>(internalField(Field::IteratedObject).get()); }
    JSCell* storage() const { return internalField(Field::Storage).get().asCell(); }
    JSSet::Helper::Entry entry() const { return JSSet::Helper::toNumber(internalField(Field::Entry).get()); }

    void setIteratedObject(VM& vm, JSSet* set) { internalField(Field::IteratedObject).set(vm, this, set); }
    void setStorage(VM& vm, JSCell* storage) { internalField(Field::Storage).set(vm, this, storage); }
    void setEntry(VM& vm, JSSet::Helper::Entry entry) { internalField(Field::Entry).set(vm, this, JSSet::Helper::toJSValue(entry)); }

private:
    JSSetIterator(VM& vm, Structure* structure)
        : Base(vm, structure)
    {
    }

    JS_EXPORT_PRIVATE void finishCreation(JSGlobalObject*, JSSet*, IterationKind);
    void finishCreation(VM&);
    DECLARE_VISIT_CHILDREN;
};
STATIC_ASSERT_IS_TRIVIALLY_DESTRUCTIBLE(JSSetIterator);

JSC_DECLARE_HOST_FUNCTION(setIteratorPrivateFuncSetIteratorNext);
JSC_DECLARE_HOST_FUNCTION(setIteratorPrivateFuncSetIteratorKey);

} // namespace JSC
