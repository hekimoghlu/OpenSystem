/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 12, 2022.
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

class JSFinalizationRegistry final : public JSInternalFieldObjectImpl<1> {
public:
    using Base = JSInternalFieldObjectImpl<1>;

    enum class Field : uint8_t { 
        Callback,
    };

    static size_t allocationSize(Checked<size_t> inlineCapacity)
    {
        ASSERT_UNUSED(inlineCapacity, inlineCapacity == 0U);
        return sizeof(JSFinalizationRegistry);
    }

    template<typename CellType, SubspaceAccess mode>
    static GCClient::IsoSubspace* subspaceFor(VM& vm)
    {
        return vm.finalizationRegistrySpace<mode>();
    }

    static std::array<JSValue, numberOfInternalFields> initialValues()
    {
        return { {
            jsNull(),
        } };
    }

    const WriteBarrier<Unknown>& internalField(Field field) const { return Base::internalField(static_cast<uint32_t>(field)); }
    WriteBarrier<Unknown>& internalField(Field field) { return Base::internalField(static_cast<uint32_t>(field)); }

    JSObject* callback() const { return jsCast<JSObject*>(internalField(Field::Callback).get()); }

    static JSFinalizationRegistry* create(VM&, Structure*, JSObject* callback);
    static JSFinalizationRegistry* createWithInitialValues(VM&, Structure*);
    static Structure* createStructure(VM&, JSGlobalObject*, JSValue prototype);

    void runFinalizationCleanup(JSGlobalObject*);

    DECLARE_EXPORT_INFO;

    void finalizeUnconditionally(VM&, CollectionScope);
    DECLARE_VISIT_CHILDREN;
    static void destroy(JSCell*);
    static constexpr DestructionMode needsDestruction = NeedsDestruction;

    JSValue takeDeadHoldingsValue();

    bool unregister(VM&, JSCell* token);
    // token should be a JSObject, Symbol, or undefined.
    void registerTarget(VM&, JSCell* target, JSValue holdings, JSValue token);

    struct LiveRegistration {
        JSCell* target;
        JSValue heldValue;
        JSCell* unregisterToken = nullptr;
    };
    JS_EXPORT_PRIVATE size_t liveCount(const Locker<JSCellLock>&);
    Vector<LiveRegistration> liveRegistrations(const Locker<JSCellLock>&) const;

    struct DeadRegistration {
        JSValue heldValue;
        JSCell* unregisterToken = nullptr;
    };
    JS_EXPORT_PRIVATE size_t deadCount(const Locker<JSCellLock>&);
    Vector<DeadRegistration> deadRegistrations(const Locker<JSCellLock>&) const;

private:
    JSFinalizationRegistry(VM& vm, Structure* structure)
        : Base(vm, structure)
    {
    }

    JS_EXPORT_PRIVATE void finishCreation(VM&, JSGlobalObject*, JSObject* callback);

    struct Registration {
        JSCell* target;
        WriteBarrier<Unknown> holdings;
    };

    using LiveRegistrations = Vector<Registration>;
    // We don't need the target anymore since we know it's dead.
    using DeadRegistrations = Vector<WriteBarrier<Unknown>>;

    // Note that we don't bother putting a write barrier on the key or target because they are weakly referenced.
    UncheckedKeyHashMap<JSCell*, LiveRegistrations> m_liveRegistrations;
    UncheckedKeyHashMap<JSCell*, DeadRegistrations> m_deadRegistrations;
    // We use a separate list for no unregister values instead of a special key in the tables above because the UncheckedKeyHashMap has a tendency to reallocate under us when iterating...
    LiveRegistrations m_noUnregistrationLive;
    DeadRegistrations m_noUnregistrationDead;
    bool m_hasAlreadyScheduledWork { false };
};

} // namespace JSC


