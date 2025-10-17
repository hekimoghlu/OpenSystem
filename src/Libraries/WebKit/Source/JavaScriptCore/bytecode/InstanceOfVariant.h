/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 31, 2025.
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

#include "ObjectPropertyConditionSet.h"
#include "StructureSet.h"
#include <wtf/TZoneMalloc.h>

namespace JSC {

class InstanceOfStatus;

class InstanceOfVariant {
    WTF_MAKE_TZONE_ALLOCATED(InstanceOfVariant);
public:
    InstanceOfVariant() = default;
    InstanceOfVariant(const StructureSet&, const ObjectPropertyConditionSet&, JSObject* prototype, bool isHit);
    
    explicit operator bool() const { return !!m_structureSet.size(); }
    
    const StructureSet& structureSet() const { return m_structureSet; }
    StructureSet& structureSet() { return m_structureSet; }
    
    const ObjectPropertyConditionSet& conditionSet() const { return m_conditionSet; }
    
    JSObject* prototype() const { return m_prototype; }
    
    bool isHit() const { return m_isHit; }
    
    bool attemptToMerge(const InstanceOfVariant& other);
    
    void dump(PrintStream&) const;
    void dumpInContext(PrintStream&, DumpContext*) const;

    bool overlaps(const InstanceOfVariant& other)
    {
        return structureSet().overlaps(other.structureSet());
    }
    
private:
    friend class InstanceOfStatus;
    
    StructureSet m_structureSet;
    ObjectPropertyConditionSet m_conditionSet;
    JSObject* m_prototype { nullptr };
    bool m_isHit { false };
};

} // namespace JSC

