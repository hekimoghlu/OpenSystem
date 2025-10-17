/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 18, 2022.
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

#include "OrderedHashTableHelper.h"

namespace JSC {

template<typename Traits>
class OrderedHashTable : public JSNonFinalObject {
    using Base = JSNonFinalObject;

public:
    using HashTable = OrderedHashTable<Traits>;
    using Helper = OrderedHashTableHelper<Traits>;
    using Storage = JSImmutableButterfly;
    using TableIndex = typename Helper::TableIndex;

    DECLARE_VISIT_CHILDREN;

    OrderedHashTable(VM& vm, Structure* structure)
        : Base(vm, structure)
    {
    }

    static ptrdiff_t offsetOfButterfly() { return OBJECT_OFFSETOF(OrderedHashTable, m_storage); }

    void finishCreation(VM& vm) { Base::finishCreation(vm); }
    void finishCreation(JSGlobalObject* globalObject, VM& vm, HashTable* base)
    {
        auto scope = DECLARE_THROW_SCOPE(vm);
        Base::finishCreation(vm);

        if (base->m_storage) {
            Storage* storage = Helper::copy(globalObject, base->storageRef());
            RETURN_IF_EXCEPTION(scope, void());
            m_storage.set(vm, this, storage);
        }
    }

    ALWAYS_INLINE JSValue* getKeySlot(JSGlobalObject* globalObject, JSValue key, uint32_t hash)
    {
        if (m_storage)
            return Helper::find(globalObject, storageRef(), key, hash).entryKeySlot;
        return nullptr;
    }

    ALWAYS_INLINE bool has(JSGlobalObject* globalObject, JSValue key)
    {
        if (m_storage) {
            auto result = Helper::find(globalObject, storageRef(), key);
            return result.entryKeyIndex != Helper::InvalidTableIndex;
        }
        return false;
    }

    ALWAYS_INLINE void add(JSGlobalObject* globalObject, JSValue key, JSValue value = { })
    {
        VM& vm = getVM(globalObject);
        auto scope = DECLARE_THROW_SCOPE(vm);

        materializeIfNeeded(globalObject);
        RETURN_IF_EXCEPTION(scope, void());

        RELEASE_AND_RETURN(scope, Helper::add(globalObject, this, storageRef(), key, value));
    }
    ALWAYS_INLINE void addNormalized(JSGlobalObject* globalObject, JSValue key, JSValue value, uint32_t hash)
    {
        VM& vm = getVM(globalObject);
        auto scope = DECLARE_THROW_SCOPE(vm);

        materializeIfNeeded(globalObject);
        RETURN_IF_EXCEPTION(scope, void());

        RELEASE_AND_RETURN(scope, Helper::addNormalized(globalObject, this, storageRef(), key, value, hash));
    }

    ALWAYS_INLINE bool remove(JSGlobalObject* globalObject, JSValue key)
    {
        if (m_storage)
            return Helper::remove(globalObject, this, storageRef(), key);
        return false;
    }
    ALWAYS_INLINE bool removeNormalized(JSGlobalObject* globalObject, JSValue key, uint32_t hash)
    {
        if (m_storage)
            return Helper::removeNormalized(globalObject, this, storageRef(), key, hash);
        return false;
    }

    ALWAYS_INLINE uint32_t size()
    {
        if (m_storage)
            return Helper::aliveEntryCount(storageRef());
        return 0;
    }

    ALWAYS_INLINE void clear(JSGlobalObject* globalObject)
    {
        if (m_storage)
            Helper::clear(globalObject, this, storageRef());
    }

    ALWAYS_INLINE void materializeIfNeeded(JSGlobalObject* globalObject)
    {
        VM& vm = getVM(globalObject);
        auto scope = DECLARE_THROW_SCOPE(vm);

        if (m_storage)
            return;

        Storage* storage = Helper::tryCreate(globalObject);
        RETURN_IF_EXCEPTION(scope, void());
        m_storage.set(vm, this, storage);
    }

    ALWAYS_INLINE JSCell* storage(JSGlobalObject* globalObject)
    {
        materializeIfNeeded(globalObject);
        ASSERT(m_storage);
        return m_storage.get();
    }

    ALWAYS_INLINE JSCell* storageOrSentinel(VM& vm)
    {
        if (m_storage)
            return m_storage.get();
        return vm.orderedHashTableSentinel();
    }

    ALWAYS_INLINE Storage& storageRef()
    {
        ASSERT(m_storage);
        return *m_storage.get();
    }

    WriteBarrier<Storage> m_storage;
};

class OrderedHashMap : public OrderedHashTable<MapTraits> {
    using Base = OrderedHashTable<MapTraits>;

public:
    OrderedHashMap(VM& vm, Structure* structure)
        : Base(vm, structure)
    {
    }

    template<typename FindKeyFunctor>
    ALWAYS_INLINE JSValue getImpl(JSGlobalObject* globalObject, const FindKeyFunctor& findKeyFunctor)
    {
        VM& vm = getVM(globalObject);
        auto scope = DECLARE_THROW_SCOPE(vm);

        if (m_storage) {
            Storage& storage = storageRef();
            auto result = findKeyFunctor(storage);
            RETURN_IF_EXCEPTION(scope, { });

            if (!Helper::isValidTableIndex(result.entryKeyIndex))
                return { };
            return Helper::get(storage, result.entryKeyIndex + 1);
        }
        return { };
    }
    ALWAYS_INLINE JSValue get(JSGlobalObject* globalObject, JSValue key)
    {
        JSValue result = getImpl(globalObject, [&](Storage& storage) ALWAYS_INLINE_LAMBDA {
            return Helper::find(globalObject, storage, key);
        });
        return result.isEmpty() ? jsUndefined() : result;
    }
    ALWAYS_INLINE JSValue get(JSGlobalObject* globalObject, JSValue key, uint32_t hash)
    {
        JSValue result = getImpl(globalObject, [&](Storage& storage) ALWAYS_INLINE_LAMBDA {
            return Helper::find(globalObject, storage, key, hash);
        });
        return result.isEmpty() ? jsUndefined() : result;
    }
    ALWAYS_INLINE JSValue get(TableIndex keyIndex)
    {
        ASSERT(m_storage);
        return Helper::get(storageRef(), keyIndex + 1);
    }

    template<typename GetValueFunctor>
    ALWAYS_INLINE JSValue getOrInsert(JSGlobalObject* globalObject, JSValue key, const GetValueFunctor& getValueFunctor)
    {
        VM& vm = getVM(globalObject);
        auto scope = DECLARE_THROW_SCOPE(vm);

        materializeIfNeeded(globalObject);
        RETURN_IF_EXCEPTION(scope, { });

        Storage& storage = storageRef();

        JSValue value;

        auto result = Helper::find(globalObject, storage, key);
        RETURN_IF_EXCEPTION(scope, { });

        if (Helper::isValidTableIndex(result.entryKeyIndex))
            value = Helper::get(storage, result.entryKeyIndex + 1);
        else {
            value = getValueFunctor();
            RETURN_IF_EXCEPTION(scope, { });

            Helper::addImpl(globalObject, this, storage, key, value, result);
            RETURN_IF_EXCEPTION(scope, { });
        }

        return value;
    }

    static JSCell* createSentinel(VM& vm) { return Helper::tryCreate(vm, 0); }
    static Symbol* createDeletedValue(VM& vm) { return Symbol::create(vm); }
};

class OrderedHashSet : public OrderedHashTable<SetTraits> {
    using Base = OrderedHashTable<SetTraits>;

public:
    OrderedHashSet(VM& vm, Structure* structure)
        : Base(vm, structure)
    {
    }
};

} // namespace JSC
