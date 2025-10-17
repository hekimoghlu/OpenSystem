/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 21, 2022.
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
#include "B3CanonicalizePrePostIncrements.h"

#include "B3BackwardsDominators.h"
#include "B3BasicBlockInlines.h"
#include "B3BlockInsertionSet.h"
#include "B3Dominators.h"
#include "B3InsertionSetInlines.h"
#include "B3PhaseScope.h"
#include "B3ProcedureInlines.h"
#include "B3ValueInlines.h"
#include <wtf/HashMap.h>
#include <wtf/IndexSet.h>

#if ENABLE(B3_JIT)

namespace JSC {
namespace B3 {

bool canonicalizePrePostIncrements(Procedure& proc)
{
    if (!isARM64())
        return false;
    PhaseScope phaseScope(proc, "canonicalizePrePostIncrements"_s);
    using Arg = Air::Arg;

    InsertionSet insertionSet { proc };
    BlockInsertionSet blockInsertionSet { proc };

    Dominators& dominators = proc.dominators();
    BackwardsDominators& backwardsDominators = proc.backwardsDominators();

    UncheckedKeyHashMap<Value*, Vector<MemoryValue*>> baseToMemories;
    UncheckedKeyHashMap<MemoryValue*, Vector<Value*>> postIndexCandidates;

    UncheckedKeyHashMap<Value*, Vector<Value*>> addressUses;
    UncheckedKeyHashMap<std::tuple<Value*, MemoryValue::OffsetType>, Vector<Value*>> baseOffsetToAddresses;
    UncheckedKeyHashMap<MemoryValue*, Vector<Value*>> preIndexCandidates;

    UncheckedKeyHashMap<Value*, unsigned> memoryToIndex;
    UncheckedKeyHashMap<BasicBlock*, UncheckedKeyHashSet<MemoryValue*>> blockToPrePostIndexCandidates;

    auto tryFindCandidates = [&](Value* value, unsigned index) ALWAYS_INLINE_LAMBDA {
        switch (value->opcode()) {
        case Load:
        case Store: {
            MemoryValue* memory = value->as<MemoryValue>();
            Value* type = value->opcode() == Load ? memory : memory->child(0);
            if (type->type() != Int32 && type->type() != Int64)
                break;

            Value* base = memory->lastChild();
            MemoryValue::OffsetType offset = memory->offset();
            if (memory->offset()) {
                // PreIndex Load/Store Pattern:
                //     address = Add(base, offset)
                //     ...
                //     memory = MemoryValue(base, offset)
                auto addresses = baseOffsetToAddresses.find({ base, offset });
                if (addresses == baseOffsetToAddresses.end())
                    break;

                for (Value* address : addresses->value) {
                    // Double check the base and offset.
                    Value* addressBase = address->child(0);
                    MemoryValue::OffsetType addressOffset = static_cast<Value::OffsetType>(address->child(1)->asIntPtr());
                    if (UNLIKELY(base != addressBase || offset != addressOffset))
                        continue;
                    // Skip the address if it's used before the memory.
                    auto uses = addressUses.find(address);
                    if (uses != addressUses.end() && uses->value.size() > 0)
                        continue;
                    preIndexCandidates.add(memory, Vector<Value*>()).iterator->value.append(address);
                    blockToPrePostIndexCandidates.add(memory->owner, UncheckedKeyHashSet<MemoryValue*>()).iterator->value.add(memory);
                }
            } else
                baseToMemories.add(base, Vector<MemoryValue*>()).iterator->value.append(memory);
            memoryToIndex.add(memory, index);
            break;
        }

        case Add: {
            Value* address = value;
            Value* base = address->child(0);
            Value* offset = address->child(1);

            if (!offset->hasIntPtr() || address->type() != Int64)
                break;
            intptr_t offset64 = offset->asIntPtr();
            Value::OffsetType offset32 = static_cast<Value::OffsetType>(offset64);
            if (offset32 != offset64 || !Arg::isValidIncrementIndexForm(offset32))
                break;

            // So far this Add value is a valid address candidate for both PreIndex and PostIndex patterns.
            addressUses.add(address, Vector<Value*>());
            baseOffsetToAddresses.add({ base, offset32 }, Vector<Value*>()).iterator->value.append(address);

            // PostIndex Load/Store Pattern:
            //     memory = MemoryValue(base, 0)
            //     ...
            //     address = Add(base, offset)
            auto memories = baseToMemories.find(base);
            if (memories == baseToMemories.end())
                break;
            for (MemoryValue* memory : memories->value) {
                postIndexCandidates.add(memory, Vector<Value*>()).iterator->value.append(address);
                blockToPrePostIndexCandidates.add(memory->owner, UncheckedKeyHashSet<MemoryValue*>()).iterator->value.add(memory);
            }
            break;
        }

        default:
            break;
        }

        for (Value* child : value->children()) {
            auto uses = addressUses.find(child);
            if (uses != addressUses.end())
                uses->value.append(value);
        }
    };

    for (BasicBlock* basicBlock : proc.blocksInPreOrder()) {
        for (unsigned index = 0; index < basicBlock->size(); ++index)
            tryFindCandidates(basicBlock->at(index), index);
    }

    auto controlEquivalent = [&](Value* v1, Value* v2) ALWAYS_INLINE_LAMBDA {
        return dominators.dominates(v1->owner, v2->owner) && backwardsDominators.dominates(v2->owner, v1->owner);
    };

    IndexSet<Value*> handledValues;
    for (const auto& entry : blockToPrePostIndexCandidates) {
        BasicBlock* block = entry.key;
        for (MemoryValue* memory : entry.value) {
            //         PreIndex Load/Store Pattern        |             Canonical Form
            // --------------------------------------------------------------------------------------
            //     address = Add(base, offset)            |    address = Nop
            //     ...                                    |    ...
            //     ...                                    |    newAddress = Add(base, offset)
            //     memory = MemoryValue(base, offset)     |    memory = MemoryValue(base, offset)
            //     ...                                    |    ...
            //     use = B3Opcode(address, ...)           |    use = B3Opcode(newAddress, ...)
            //
            auto tryPreIndexTransform = [&]() ALWAYS_INLINE_LAMBDA {
                for (Value* address : preIndexCandidates.get(memory)) {
                    if (handledValues.contains(address) || !controlEquivalent(address, memory))
                        continue;

                    // We need to do this because we have to move the address to the newAddress,
                    // which is located just before the memory.
                    auto uses = addressUses.find(address);
                    ASSERT(uses != addressUses.end() && uses->value.size());
                    for (Value* use : uses->value) {
                        if (!dominators.dominates(memory->owner, use->owner))
                            continue;
                    }

                    unsigned index = memoryToIndex.get(memory);
                    Value* newAddress = insertionSet.insert<Value>(index, Add, memory->origin(), address->child(0), address->child(1));
                    for (Value* use : addressUses.get(address)) {
                        for (unsigned i = 0; i < use->numChildren(); ++i) {
                            Value*& child = use->child(i);
                            if (child == address)
                                child = newAddress;
                        }
                    }
                    address->replaceWithNopIgnoringType();

                    handledValues.add(address);
                    return true;
                }
                return false;
            };

            if (tryPreIndexTransform())
                continue;

            //      PostIndex Load/Store Pattern     |            Canonical Form
            // -------------------------------------------------------------------------------
            //     ...                               |    newOffset = Constant
            //     ...                               |    newAddress = Add(base, newOffset)
            //     memory = MemoryValue(base, 0)     |    memory = MemoryValue(base, 0)
            //     ...                               |    ...
            //     address = Add(base, offset)       |    address = Identity(newAddress)
            //
            for (Value* address : postIndexCandidates.get(memory)) {
                if (handledValues.contains(address) || !controlEquivalent(memory, address))
                    continue;

                unsigned index = memoryToIndex.get(memory);
                Value* newOffset = insertionSet.insert<Const64Value>(index, memory->origin(), address->child(1)->asInt());
                Value* newAddress = insertionSet.insert<Value>(index, Add, memory->origin(), address->child(0), newOffset);
                address->replaceWithIdentity(newAddress);

                handledValues.add(address);
            }
        }
        // This will reset the indexes of values if there are any new insertions.
        insertionSet.execute(block);
    }
    return true;
}

}
} // namespace JSC::B3

#endif // ENABLE(B3_JIT)
