/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 1, 2023.
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
#include "StructureStubClearingWatchpoint.h"

#if ENABLE(JIT)

#include "CodeBlockInlines.h"
#include "JSCellInlines.h"
#include "StructureStubInfo.h"
#include <wtf/TZoneMallocInlines.h>

namespace JSC {

WTF_MAKE_TZONE_ALLOCATED_IMPL(StructureStubInfoClearingWatchpoint);
WTF_MAKE_TZONE_ALLOCATED_IMPL(AdaptiveValueStructureStubClearingWatchpoint);
WTF_MAKE_TZONE_ALLOCATED_IMPL(StructureTransitionStructureStubClearingWatchpoint);

StructureStubInfoClearingWatchpoint::~StructureStubInfoClearingWatchpoint()
{
    ASSERT(!m_owner->wasDestructed());
}

void StructureStubInfoClearingWatchpoint::fireInternal(VM&, const FireDetail&)
{
    ASSERT(!m_owner->wasDestructed());
    if (m_owner->isPendingDestruction())
        return;

    // This will implicitly cause my own demise: stub reset removes all watchpoints.
    // That works, because deleting a watchpoint removes it from the set's list, and
    // the set's list traversal for firing is robust against the set changing.
    ConcurrentJSLocker locker(m_owner->m_lock);
    m_stubInfo.reset(locker, m_owner.get());
}

void StructureTransitionStructureStubClearingWatchpoint::fireInternal(VM& vm, const FireDetail&)
{
    if (m_owner->ownerIsDead())
        return;

    if (!m_key || !m_key.isWatchable(PropertyCondition::EnsureWatchability)) {
        StringFireDetail detail("IC has been invalidated");
        Ref { m_watchpointSet }->fireAll(vm, detail);
        return;
    }

    if (m_key.kind() == PropertyCondition::Presence) {
        // If this was a presence condition, let's watch the property for replacements. This is profitable
        // for the DFG, which will want the replacement set to be valid in order to do constant folding.
        m_key.object()->structure()->startWatchingPropertyForReplacements(vm, m_key.offset());
    }

    m_key.object()->structure()->addTransitionWatchpoint(this);
}

void AdaptiveValueStructureStubClearingWatchpoint::handleFire(VM& vm, const FireDetail&)
{
    if (m_owner->ownerIsDead())
        return;

    StringFireDetail detail("IC has been invalidated");
    Ref { m_watchpointSet }->fireAll(vm, detail);
}

} // namespace JSC

#endif // ENABLE(JIT)

