/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 5, 2024.
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
#include "PolyProtoAccessChain.h"

#include "CacheableIdentifierInlines.h"
#include "JSCInlines.h"

namespace JSC {

RefPtr<PolyProtoAccessChain> PolyProtoAccessChain::tryCreate(JSGlobalObject* globalObject, JSCell* base, CacheableIdentifier propertyName, const PropertySlot& slot)
{
    JSObject* target = slot.isUnset() ? nullptr : slot.slotBase();
    return tryCreate(globalObject, base, propertyName, target);
}

RefPtr<PolyProtoAccessChain> PolyProtoAccessChain::tryCreate(JSGlobalObject* globalObject, JSCell* base, CacheableIdentifier propertyName, JSObject* target)
{
    JSCell* current = base;

    bool found = false;

    Vector<StructureID> chain;
    for (unsigned iterationNumber = 0; true; ++iterationNumber) {
        Structure* structure = current->structure();

        if (structure->isDictionary())
            return nullptr;

        if (!structure->propertyAccessesAreCacheable())
            return nullptr;

        if (structure->isProxy())
            return nullptr;

        // To save memory, we don't include the base in the chain. We let
        // AccessCase provide the base to us as needed.
        if (iterationNumber)
            chain.append(structure->id());
        else
            RELEASE_ASSERT(current == base);

        if (current == target) {
            found = true;
            break;
        }

        // TypedArray has an ability to stop [[Prototype]] traversing for numeric index string (e.g. "0.1").
        // If we found it, then traverse should stop for Unset case.
        // https://262.ecma-international.org/9.0/#_ref_2826
        if (!target && isTypedArrayType(structure->typeInfo().type()) && isCanonicalNumericIndexString(propertyName.uid())) {
            found = true;
            break;
        }

        JSValue prototype = structure->prototypeForLookup(globalObject, current);
        if (prototype.isNull())
            break;
        current = asObject(prototype);
    }

    if (!found && !!target)
        return nullptr;

    return adoptRef(*new PolyProtoAccessChain(WTFMove(chain)));
}

bool PolyProtoAccessChain::needImpurePropertyWatchpoint(VM&) const
{
    for (StructureID structureID : m_chain) {
        if (structureID.decode()->needImpurePropertyWatchpoint())
            return true;
    }
    return false;
}

bool PolyProtoAccessChain::operator==(const PolyProtoAccessChain& other) const
{
    return m_chain == other.m_chain;
}

void PolyProtoAccessChain::dump(Structure* baseStructure, PrintStream& out) const
{
    out.print("PolyPolyProtoAccessChain: [\n");
    forEach(baseStructure->vm(), baseStructure, [&] (Structure* structure, bool) {
        out.print("\t");
        structure->dump(out);
        out.print("\n");
    });
}

}
