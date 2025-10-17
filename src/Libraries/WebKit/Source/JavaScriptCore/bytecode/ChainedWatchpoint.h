/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 2, 2025.
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

#include "PackedCellPtr.h"
#include "Watchpoint.h"

namespace JSC {

class ChainedWatchpoint final : public Watchpoint {
public:
    ChainedWatchpoint(JSCell* owner, InlineWatchpointSet& watchpointSet)
        : Watchpoint(Watchpoint::Type::Chained)
        , m_owner(owner)
        , m_watchpointSet(watchpointSet)
    {
        RELEASE_ASSERT(watchpointSet.state() == IsWatched);
    }

    void install(InlineWatchpointSet& fromWatchpoint, VM&);

    void fireInternal(VM&, const FireDetail&);

private:
    PackedCellPtr<JSCell> m_owner;
    InlineWatchpointSet& m_watchpointSet;
};

inline void ChainedWatchpoint::install(InlineWatchpointSet& fromWatchpoint, VM&)
{
    fromWatchpoint.add(this);
}

inline void ChainedWatchpoint::fireInternal(VM& vm, const FireDetail&)
{
    if (!m_owner->isPendingDestruction())
        m_watchpointSet.fireAll(vm, StringFireDetail("chained watchpoint is fired."));
}

} // namespace JSC
