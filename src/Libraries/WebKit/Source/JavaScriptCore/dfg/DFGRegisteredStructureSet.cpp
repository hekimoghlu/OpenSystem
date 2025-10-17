/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 14, 2021.
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
#include "DFGRegisteredStructureSet.h"

#if ENABLE(DFG_JIT)

#include "DFGAbstractValue.h"
#include "TrackedReferences.h"

namespace JSC { namespace DFG {

void RegisteredStructureSet::filter(const DFG::StructureAbstractValue& other)
{
    genericFilter(
        [&] (RegisteredStructure structure) -> bool {
            return other.contains(structure); 
        });
}

void RegisteredStructureSet::filter(SpeculatedType type)
{
    genericFilter(
        [&] (RegisteredStructure structure) -> bool {
            return type & speculationFromStructure(structure.get());
        });
}

void RegisteredStructureSet::filterArrayModes(ArrayModes arrayModes)
{
    genericFilter(
        [&] (RegisteredStructure structure) -> bool {
            return arrayModes & arrayModesFromStructure(structure.get());
        });
}

void RegisteredStructureSet::filter(const DFG::AbstractValue& other)
{
    filter(other.m_structure);
    filter(other.m_type);
    filterArrayModes(other.m_arrayModes);
}

SpeculatedType RegisteredStructureSet::speculationFromStructures() const
{
    SpeculatedType result = SpecNone;
    forEach(
        [&] (RegisteredStructure structure) {
            mergeSpeculation(result, speculationFromStructure(structure.get()));
        });
    return result;
}

ArrayModes RegisteredStructureSet::arrayModesFromStructures() const
{
    ArrayModes result = 0;
    forEach(
        [&] (RegisteredStructure structure) {
            mergeArrayModes(result, arrayModesFromStructure(structure.get()));
        });
    return result;
}

void RegisteredStructureSet::validateReferences(const TrackedReferences& trackedReferences) const
{
    // The type system should help us here, but protect people from getting that wrong using std::bit_cast or something crazy.
    forEach(
        [&] (RegisteredStructure structure) {
            trackedReferences.check(structure.get());
        });
}

} } // namespace JSC::DFG

#endif // ENABLE(DFG_JIT)
