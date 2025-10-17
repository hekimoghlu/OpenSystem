/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 22, 2022.
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

#if ENABLE(DFG_JIT)

#include "ArrayProfile.h"
#include "DFGRegisteredStructure.h"
#include "StructureSet.h"
#include <wtf/TinyPtrSet.h>

namespace JSC {

class TrackedReferences;

namespace DFG {

struct AbstractValue;
class StructureAbstractValue;

class RegisteredStructureSet : public TinyPtrSet<RegisteredStructure> {
public:

    RegisteredStructureSet()
    { }
    
    RegisteredStructureSet(RegisteredStructure structure)
        : TinyPtrSet(structure)
    {
    }
    
    RegisteredStructure onlyStructure() const
    {
        return onlyEntry();
    }

    StructureSet toStructureSet() const
    {
        StructureSet result;
        forEach([&] (RegisteredStructure structure) { result.add(structure.get()); });
        return result;
    }

    void filter(const DFG::StructureAbstractValue&);
    void filter(SpeculatedType);
    void filterArrayModes(ArrayModes);
    void filter(const DFG::AbstractValue&);
    
    SpeculatedType speculationFromStructures() const;
    ArrayModes arrayModesFromStructures() const;

    void validateReferences(const TrackedReferences&) const;
};

} } // namespace JSC::DFG

#endif // ENABLE(DFG_JIT)
