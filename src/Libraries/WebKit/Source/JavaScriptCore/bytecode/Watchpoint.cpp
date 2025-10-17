/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 3, 2022.
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
#include "Watchpoint.h"

#include "AdaptiveInferredPropertyValueWatchpointBase.h"
#include "CachedSpecialPropertyAdaptiveStructureWatchpoint.h"
#include "ChainedWatchpoint.h"
#include "CodeBlockJettisoningWatchpoint.h"
#include "DFGAdaptiveStructureWatchpoint.h"
#include "FunctionRareData.h"
#include "HeapInlines.h"
#include "LLIntPrototypeLoadAdaptiveStructureWatchpoint.h"
#include "ObjectAdaptiveStructureWatchpoint.h"
#include "StructureRareDataInlines.h"
#include "StructureStubClearingWatchpoint.h"
#include "VM.h"

namespace JSC {

DEFINE_ALLOCATOR_WITH_HEAP_IDENTIFIER(Watchpoint);
DEFINE_ALLOCATOR_WITH_HEAP_IDENTIFIER(WatchpointSet);

void StringFireDetail::dump(PrintStream& out) const
{
    out.print(m_string);
}

template<typename Func>
inline void Watchpoint::runWithDowncast(const Func& func)
{
    switch (m_type) {
#define JSC_DEFINE_WATCHPOINT_DISPATCH(type, cast) \
    case Type::type: \
        func(static_cast<cast*>(this)); \
        break;
    JSC_WATCHPOINT_TYPES(JSC_DEFINE_WATCHPOINT_DISPATCH)
#undef JSC_DEFINE_WATCHPOINT_DISPATCH
    }
}

void Watchpoint::operator delete(Watchpoint* watchpoint, std::destroying_delete_t)
{
    watchpoint->runWithDowncast([](auto* derived) {
        std::destroy_at(derived);
        std::decay_t<decltype(*derived)>::freeAfterDestruction(derived);
    });
}

Watchpoint::~Watchpoint()
{
    if (isOnList()) {
        // This will happen if we get destroyed before the set fires. That's totally a valid
        // possibility. For example:
        //
        // CodeBlock has a Watchpoint on transition from structure S1. The transition never
        // happens, but the CodeBlock gets destroyed because of GC.
        remove();
    }
}

void Watchpoint::fire(VM& vm, const FireDetail& detail)
{
    RELEASE_ASSERT(!isOnList());
    runWithDowncast([&](auto* derived) {
        derived->fireInternal(vm, detail);
    });
}

WatchpointSet::WatchpointSet(WatchpointState state)
    : m_state(state)
    , m_setIsNotEmpty(false)
{
}

WatchpointSet::~WatchpointSet()
{
    // Remove all watchpoints, so that they don't try to remove themselves. Note that we
    // don't fire watchpoints on deletion. We assume that any code that is interested in
    // watchpoints already also separately has a mechanism to make sure that the code is
    // either keeping the watchpoint set's owner alive, or does some weak reference thing.
    while (!m_set.isEmpty())
        m_set.begin()->remove();
}

void WatchpointSet::add(Watchpoint* watchpoint)
{
    ASSERT(!isCompilationThread());
    ASSERT(state() != IsInvalidated);
    if (!watchpoint)
        return;
    m_set.push(watchpoint);
    m_setIsNotEmpty = true;
    m_state = IsWatched;
}

void WatchpointSet::fireAllSlow(VM& vm, const FireDetail& detail)
{
    ASSERT(state() == IsWatched);
    
    WTF::storeStoreFence();
    m_state = IsInvalidated; // Do this first. Needed for adaptive watchpoints.
    fireAllWatchpoints(vm, detail);
    WTF::storeStoreFence();
}

void WatchpointSet::fireAllSlow(VM&, DeferredWatchpointFire* deferredWatchpoints)
{
    ASSERT(state() == IsWatched);

    WTF::storeStoreFence();
    deferredWatchpoints->takeWatchpointsToFire(this);
    m_state = IsInvalidated; // Do after moving watchpoints to deferredWatchpoints so deferredWatchpoints gets our current state.
    WTF::storeStoreFence();
}

void WatchpointSet::fireAllSlow(VM& vm, const char* reason)
{
    fireAllSlow(vm, StringFireDetail(reason));
}

void WatchpointSet::fireAllWatchpoints(VM& vm, const FireDetail& detail)
{
    // In case there are any adaptive watchpoints, we need to make sure that they see that this
    // watchpoint has been already invalidated.
    RELEASE_ASSERT(hasBeenInvalidated());

    // Firing a watchpoint may cause a GC to happen. This GC could destroy various
    // Watchpoints themselves while they're in the process of firing. It's not safe
    // for most Watchpoints to be destructed while they're in the middle of firing.
    // This GC could also destroy us, and we're not in a safe state to be destroyed.
    // The safest thing to do is to DeferGCForAWhile to prevent this GC from happening.
    DeferGCForAWhile deferGC(vm);
    
    while (!m_set.isEmpty()) {
        Watchpoint& watchpoint = *m_set.begin();
        ASSERT(watchpoint.isOnList());
        
        // Removing the Watchpoint before firing it makes it possible to implement watchpoints
        // that add themselves to a different set when they fire. This kind of "adaptive"
        // watchpoint can be used to track some semantic property that is more fine-graiend than
        // what the set can convey. For example, we might care if a singleton object ever has a
        // property called "foo". We can watch for this by checking if its Structure has "foo" and
        // then watching its transitions. But then the watchpoint fires if any property is added.
        // So, before the watchpoint decides to invalidate any code, it can check if it is
        // possible to add itself to the transition watchpoint set of the singleton object's new
        // Structure.
        watchpoint.remove();
        ASSERT(&*m_set.begin() != &watchpoint);
        ASSERT(!watchpoint.isOnList());
        
        watchpoint.fire(vm, detail);
        // After we fire the watchpoint, the watchpoint pointer may be a dangling pointer. That's
        // fine, because we have no use for the pointer anymore.
    }
}

void WatchpointSet::take(WatchpointSet* other)
{
    ASSERT(state() == ClearWatchpoint);
    m_set.takeFrom(other->m_set);
    m_setIsNotEmpty = other->m_setIsNotEmpty;
    m_state = other->m_state;
    other->m_setIsNotEmpty = false;
}

void InlineWatchpointSet::add(Watchpoint* watchpoint)
{
    inflate()->add(watchpoint);
}

void InlineWatchpointSet::fireAll(VM& vm, const char* reason)
{
    fireAll(vm, StringFireDetail(reason));
}

WatchpointSet* InlineWatchpointSet::inflateSlow()
{
    ASSERT(isThin());
    ASSERT(!isCompilationThread());
    WatchpointSet* fat = &WatchpointSet::create(decodeState(m_data)).leakRef();
    WTF::storeStoreFence();
    m_data = std::bit_cast<uintptr_t>(fat);
    return fat;
}

void InlineWatchpointSet::freeFat()
{
    ASSERT(isFat());
    fat()->deref();
}

void DeferredWatchpointFire::takeWatchpointsToFire(WatchpointSet* watchpointsToFire)
{
    ASSERT(m_watchpointsToFire.state() == ClearWatchpoint);
    ASSERT(watchpointsToFire->state() == IsWatched);
    m_watchpointsToFire.take(watchpointsToFire);
}

} // namespace JSC

namespace WTF {

void printInternal(PrintStream& out, JSC::WatchpointState state)
{
    switch (state) {
    case JSC::ClearWatchpoint:
        out.print("ClearWatchpoint");
        return;
    case JSC::IsWatched:
        out.print("IsWatched");
        return;
    case JSC::IsInvalidated:
        out.print("IsInvalidated");
        return;
    }
    RELEASE_ASSERT_NOT_REACHED();
}

} // namespace WTF

