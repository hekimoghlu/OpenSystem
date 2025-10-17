/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 11, 2022.
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
#include "JSFinalizationRegistry.h"

#include "AbstractSlotVisitor.h"
#include "DeferredWorkTimer.h"
#include "GlobalObjectMethodTable.h"
#include "JSCInlines.h"
#include "JSInternalFieldObjectImplInlines.h"

namespace JSC {

const ClassInfo JSFinalizationRegistry::s_info = { "FinalizationRegistry"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(JSFinalizationRegistry) };

Structure* JSFinalizationRegistry::createStructure(VM& vm, JSGlobalObject* globalObject, JSValue prototype)
{
    return Structure::create(vm, globalObject, prototype, TypeInfo(ObjectType, StructureFlags), info());
}

JSFinalizationRegistry* JSFinalizationRegistry::create(VM& vm, Structure* structure, JSObject* callback)
{
    JSFinalizationRegistry* instance = new (NotNull, allocateCell<JSFinalizationRegistry>(vm)) JSFinalizationRegistry(vm, structure);
    instance->finishCreation(vm, structure->globalObject(), callback);
    return instance;
}

void JSFinalizationRegistry::finishCreation(VM& vm, JSGlobalObject* globalObject, JSObject* callback)
{
    Base::finishCreation(vm);
    ASSERT(callback->isCallable());
    auto values = initialValues();
    for (unsigned index = 0; index < values.size(); ++index)
        Base::internalField(index).setWithoutWriteBarrier(values[index]);
    internalField(Field::Callback).setWithoutWriteBarrier(callback);

    // Make sure we init the DOM wrapper for our document since it must be allocated before finalizeUnconditionally is called. finalizeUnconditionally,
    // is called during the GC flip so no JS objects can be allocated there. This only works because we no longer weakly hold on to DOM wrappers.
    globalObject->globalObjectMethodTable()->currentScriptExecutionOwner(globalObject);
}

template<typename Visitor>
void JSFinalizationRegistry::visitChildrenImpl(JSCell* cell, Visitor& visitor)
{
    Base::visitChildren(cell, visitor);

    auto* thisObject = jsCast<JSFinalizationRegistry*>(cell);

    Locker locker { thisObject->cellLock() };
    for (const auto& iter : thisObject->m_liveRegistrations) {
        for (auto& registration : iter.value)
            visitor.append(registration.holdings);
    }
    for (auto& registration : thisObject->m_noUnregistrationLive)
        visitor.append(registration.holdings);
    for (const auto& iter : thisObject->m_deadRegistrations) {
        for (auto& holdings : iter.value)
            visitor.append(holdings);
    }
    for (auto& holdings : thisObject->m_noUnregistrationDead)
        visitor.append(holdings);

    size_t totalBufferSizesInBytes = thisObject->m_deadRegistrations.capacity() * sizeof(typename decltype(thisObject->m_deadRegistrations)::KeyValuePairType);
    totalBufferSizesInBytes += thisObject->m_liveRegistrations.capacity() * sizeof(typename decltype(thisObject->m_deadRegistrations)::KeyValuePairType);
    totalBufferSizesInBytes += thisObject->m_noUnregistrationLive.capacity() * sizeof(decltype(thisObject->m_noUnregistrationLive.takeLast()));
    totalBufferSizesInBytes += thisObject->m_noUnregistrationDead.capacity() * sizeof(decltype(thisObject->m_noUnregistrationLive.takeLast()));
    visitor.vm().heap.reportExtraMemoryVisited(totalBufferSizesInBytes);
}

DEFINE_VISIT_CHILDREN(JSFinalizationRegistry);

void JSFinalizationRegistry::destroy(JSCell* table)
{
    static_cast<JSFinalizationRegistry*>(table)->~JSFinalizationRegistry();
}

void JSFinalizationRegistry::finalizeUnconditionally(VM& vm, CollectionScope)
{
    Locker locker { cellLock() };

#if ASSERT_ENABLED
    for (const auto& iter : m_deadRegistrations)
        RELEASE_ASSERT(iter.value.size());
#endif

    bool readiedCell = false;
    m_noUnregistrationLive.removeAllMatching([&] (const Registration& reg) {
        ASSERT(!reg.holdings.get().isCell() || vm.heap.isMarked(reg.holdings.get().asCell()));
        if (!vm.heap.isMarked(reg.target)) {
            m_noUnregistrationDead.append(reg.holdings);
            readiedCell = true;
            return true;
        }
        return false;
    });

    m_liveRegistrations.removeIf([&] (auto& bucket) -> bool {
        ASSERT(bucket.value.size());

        bool keyIsDead = !vm.heap.isMarked(bucket.key);
        DeadRegistrations* deadList = nullptr;
        auto getDeadList = [&] () -> DeadRegistrations& {
            if (UNLIKELY(!deadList))
                deadList = &m_deadRegistrations.add(bucket.key, DeadRegistrations()).iterator->value;
            return *deadList;
        };

        bucket.value.removeAllMatching([&] (const Registration& reg) {
            ASSERT(!reg.holdings.get().isCell() || vm.heap.isMarked(reg.holdings.get().asCell()));
            if (!vm.heap.isMarked(reg.target)) {
                if (keyIsDead)
                    m_noUnregistrationDead.append(reg.holdings);
                else
                    getDeadList().append(reg.holdings);
                readiedCell = true;
                return true;
            }

            if (keyIsDead) {
                m_noUnregistrationLive.append(reg);
                return true;
            }

            return false;
        });

        return !bucket.value.size();
    });

    if (!m_hasAlreadyScheduledWork && (readiedCell || deadCount(locker))) {
        auto ticket = vm.deferredWorkTimer->addPendingWork(DeferredWorkTimer::WorkType::ImminentlyScheduled, vm, this, { });
        ASSERT(vm.deferredWorkTimer->hasPendingWork(ticket));
        vm.deferredWorkTimer->scheduleWorkSoon(ticket, [this](DeferredWorkTimer::Ticket) {
            JSGlobalObject* globalObject = this->globalObject();
            this->m_hasAlreadyScheduledWork = false;
            this->runFinalizationCleanup(globalObject);
        });
        m_hasAlreadyScheduledWork = true;
    }
}

void JSFinalizationRegistry::runFinalizationCleanup(JSGlobalObject* globalObject)
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    while (JSValue value = takeDeadHoldingsValue()) {
        MarkedArgumentBuffer args;
        args.append(value);
        call(globalObject, callback(), args, "This should not be visible: please report a bug to bugs.webkit.org"_s);
        RETURN_IF_EXCEPTION(scope, void());
    }
}

JSValue JSFinalizationRegistry::takeDeadHoldingsValue()
{
    Locker locker { cellLock() };
    JSValue result;
    if (m_noUnregistrationDead.size())
        result = m_noUnregistrationDead.takeLast().get();
    else {
        auto iter = m_deadRegistrations.begin();
        if (iter == m_deadRegistrations.end())
            return JSValue();
        ASSERT(iter->value.size());
        result = iter->value.takeLast().get();
        if (!iter->value.size())
            m_deadRegistrations.remove(iter);
    }

#if ASSERT_ENABLED
    for (const auto& iter : m_deadRegistrations)
        RELEASE_ASSERT(iter.value.size());
#endif

    return result;
}

void JSFinalizationRegistry::registerTarget(VM& vm, JSCell* target, JSValue holdings, JSValue token)
{
    Locker locker { cellLock() };
    Registration registration;
    registration.target = target;
    registration.holdings.setWithoutWriteBarrier(holdings);
    if (token.isUndefined())
        m_noUnregistrationLive.append(WTFMove(registration));
    else {
        RELEASE_ASSERT(token.isCell());
        auto result = m_liveRegistrations.add(token.asCell(), LiveRegistrations());
        result.iterator->value.append(WTFMove(registration));
    }
    vm.writeBarrier(this);
}

bool JSFinalizationRegistry::unregister(VM&, JSCell* token)
{
    // We don't need to write barrier ourselves here because we will only point to less things after this finishes.
    Locker locker { cellLock() };
    bool result = m_liveRegistrations.remove(token);
    result |= m_deadRegistrations.remove(token);

    return result;
}

size_t JSFinalizationRegistry::liveCount(const Locker<JSCellLock>&)
{
    size_t count = m_noUnregistrationLive.size();
    for (const auto& iter : m_liveRegistrations)
        count += iter.value.size();

    return count;
}

Vector<JSFinalizationRegistry::LiveRegistration> JSFinalizationRegistry::liveRegistrations(const Locker<JSCellLock>&) const
{
    Vector<LiveRegistration> liveRegistrations;

    for (const auto& registration : m_noUnregistrationLive)
        liveRegistrations.append({ registration.target, registration.holdings.get() });

    for (const auto& [unregisterToken, registrations] : m_liveRegistrations) {
        for (const auto& registration : registrations)
            liveRegistrations.append({ registration.target, registration.holdings.get(), unregisterToken });
    }

    return liveRegistrations;
}

size_t JSFinalizationRegistry::deadCount(const Locker<JSCellLock>&)
{
    size_t count = m_noUnregistrationDead.size();
    for (const auto& iter : m_deadRegistrations)
        count += iter.value.size();

    return count;
}

Vector<JSFinalizationRegistry::DeadRegistration> JSFinalizationRegistry::deadRegistrations(const Locker<JSCellLock>&) const
{
    Vector<DeadRegistration> deadRegistrations;

    for (const auto& heldValue : m_noUnregistrationDead)
        deadRegistrations.append({ heldValue.get() });

    for (const auto& [unregisterToken, heldValues] : m_deadRegistrations) {
        for (const auto& heldValue : heldValues)
            deadRegistrations.append({ heldValue.get(), unregisterToken });
    }

    return deadRegistrations;
}

}


