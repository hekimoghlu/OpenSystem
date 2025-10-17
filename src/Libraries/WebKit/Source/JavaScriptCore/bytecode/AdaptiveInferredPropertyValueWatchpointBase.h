/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 30, 2024.
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
#include "Watchpoint.h"
#include <wtf/Noncopyable.h>
#include <wtf/TZoneMalloc.h>

namespace JSC {

// FIXME: This isn't actually a Watchpoint. We should probably have a name which better reflects that:
// https://bugs.webkit.org/show_bug.cgi?id=202381
class AdaptiveInferredPropertyValueWatchpointBase {
    WTF_MAKE_NONCOPYABLE(AdaptiveInferredPropertyValueWatchpointBase);
    WTF_MAKE_TZONE_ALLOCATED(AdaptiveInferredPropertyValueWatchpointBase);

public:
    AdaptiveInferredPropertyValueWatchpointBase(const ObjectPropertyCondition&);
    AdaptiveInferredPropertyValueWatchpointBase() = default;

    const ObjectPropertyCondition& key() const { return m_key; }

    void initialize(const ObjectPropertyCondition&);
    void install(VM&);

    virtual ~AdaptiveInferredPropertyValueWatchpointBase() = default;

    class StructureWatchpoint final : public Watchpoint {
    public:
        StructureWatchpoint()
            : Watchpoint(Watchpoint::Type::AdaptiveInferredPropertyValueStructure)
        { }

        void fireInternal(VM&, const FireDetail&);
    };
    static_assert(sizeof(StructureWatchpoint) == sizeof(Watchpoint));

    class PropertyWatchpoint final : public Watchpoint {
    public:
        PropertyWatchpoint()
            : Watchpoint(Watchpoint::Type::AdaptiveInferredPropertyValueProperty)
        { }

        void fireInternal(VM&, const FireDetail&);
    };
    static_assert(sizeof(PropertyWatchpoint) == sizeof(Watchpoint));

protected:
    virtual bool isValid() const;
    virtual void handleFire(VM&, const FireDetail&) = 0;

private:
    void fire(VM&, const FireDetail&);

    ObjectPropertyCondition m_key;
    StructureWatchpoint m_structureWatchpoint;
    PropertyWatchpoint m_propertyWatchpoint;
};

} // namespace JSC
