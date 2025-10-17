/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 15, 2023.
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

#include "ObjectPropertyCondition.h"
#include "PackedCellPtr.h"
#include "Watchpoint.h"

namespace JSC {

class ObjectAdaptiveStructureWatchpoint final : public Watchpoint {
public:
    ObjectAdaptiveStructureWatchpoint(JSCell* owner, const ObjectPropertyCondition& key, InlineWatchpointSet& watchpointSet)
        : Watchpoint(Watchpoint::Type::ObjectAdaptiveStructure)
        , m_owner(owner)
        , m_key(key)
        , m_watchpointSet(watchpointSet)
    {
        RELEASE_ASSERT(key.kind() != PropertyCondition::Equivalence);
        RELEASE_ASSERT(key.watchingRequiresStructureTransitionWatchpoint());
        RELEASE_ASSERT(!key.watchingRequiresReplacementWatchpoint());
        RELEASE_ASSERT(watchpointSet.state() == IsWatched);
    }

    const ObjectPropertyCondition& key() const { return m_key; }

    void install(VM&);

    void fireInternal(VM&, const FireDetail&);

private:
    PackedCellPtr<JSCell> m_owner;
    ObjectPropertyCondition m_key;
    InlineWatchpointSet& m_watchpointSet;
};

inline void ObjectAdaptiveStructureWatchpoint::install(VM&)
{
    RELEASE_ASSERT(m_key.isWatchable(PropertyCondition::MakeNoChanges));

    m_key.object()->structure()->addTransitionWatchpoint(this);
}

inline void ObjectAdaptiveStructureWatchpoint::fireInternal(VM& vm, const FireDetail&)
{
    if (m_owner->isPendingDestruction())
        return;

    if (m_key.isWatchable(PropertyCondition::EnsureWatchability)) {
        install(vm);
        return;
    }

    m_watchpointSet.fireAll(vm, StringFireDetail("Object Property is added."));
}

} // namespace JSC
