/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 24, 2025.
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

#include "AdaptiveInferredPropertyValueWatchpointBase.h"
#include <wtf/TZoneMalloc.h>

namespace JSC {

template<typename WatchpointSet>
class ObjectPropertyChangeAdaptiveWatchpoint final : public AdaptiveInferredPropertyValueWatchpointBase {
    WTF_MAKE_TZONE_ALLOCATED_TEMPLATE(ObjectPropertyChangeAdaptiveWatchpoint);
public:
    using Base = AdaptiveInferredPropertyValueWatchpointBase;
    ObjectPropertyChangeAdaptiveWatchpoint(JSCell* owner, const ObjectPropertyCondition& condition, WatchpointSet& watchpointSet)
        : Base(condition)
        , m_owner(owner)
        , m_watchpointSet(watchpointSet)
    {
        RELEASE_ASSERT(watchpointSet.state() == IsWatched);
    }

private:
    bool isValid() const final
    {
        return !m_owner->isPendingDestruction();
    }

    void handleFire(VM& vm, const FireDetail&) final
    {
        m_watchpointSet.fireAll(vm, StringFireDetail("Object Property is changed."));
    }

    JSCell* m_owner;
    WatchpointSet& m_watchpointSet;
};

WTF_MAKE_TZONE_ALLOCATED_TEMPLATE_IMPL(template<typename WatchpointSet>, ObjectPropertyChangeAdaptiveWatchpoint<WatchpointSet>);

} // namespace JSC
