/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 10, 2024.
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
#include "B3FoldPathConstants.h"

#if ENABLE(B3_JIT)

#include "B3BasicBlockInlines.h"
#include "B3CaseCollectionInlines.h"
#include "B3Dominators.h"
#include "B3InsertionSet.h"
#include "B3PhaseScope.h"
#include "B3SwitchValue.h"
#include "B3ValueInlines.h"

namespace JSC { namespace B3 {

namespace {

namespace B3FoldPathConstantsInternal {
static constexpr bool verbose = false;
}

class FoldPathConstants {
public:
    FoldPathConstants(Procedure& proc)
        : m_proc(proc)
        , m_insertionSet(proc)
    {
    }

    void run()
    {
        bool changed = false;

        if (B3FoldPathConstantsInternal::verbose)
            dataLog("B3 before folding path constants: \n", m_proc, "\n");
        
        // Find all of the values that are the subject of a branch or switch. For any successor
        // that we dominate, install a value override at that block.

        UncheckedKeyHashMap<Value*, Vector<Override>> overrides;

        Dominators& dominators = m_proc.dominators();
        
        auto addOverride = [&] (
            BasicBlock* from, Value* value, const Override& override) {

            if (override.block->numPredecessors() != 1)
                return;
            ASSERT(override.block->predecessor(0) == from);

            Vector<Override>& forValue =
                overrides.add(value, Vector<Override>()).iterator->value;

            if (ASSERT_ENABLED) {
                for (const Override& otherOverride : forValue)
                    ASSERT_UNUSED(otherOverride, otherOverride.block != override.block);
            }

            if (B3FoldPathConstantsInternal::verbose)
                dataLog("Overriding ", *value, " from ", *from, ": ", override, "\n");
            
            forValue.append(override);
        };
        
        for (BasicBlock* block : m_proc) {
            Value* branch = block->last();
            switch (branch->opcode()) {
            case Branch:
                if (block->successorBlock(0) == block->successorBlock(1))
                    continue;
                addOverride(
                    block, branch->child(0),
                    Override::nonZero(block->successorBlock(0)));
                addOverride(
                    block, branch->child(0),
                    Override::constant(block->successorBlock(1), 0));
                break;
            case Switch: {
                SwitchValue* switchValue = branch->as<SwitchValue>();

                UncheckedKeyHashMap<BasicBlock*, unsigned> targetUses;
                for (SwitchCase switchCase : switchValue->cases(block))
                    targetUses.add(switchCase.targetBlock(), 0).iterator->value++;
                targetUses.add(switchValue->fallThrough(block), 0).iterator->value++;

                for (SwitchCase switchCase : switchValue->cases(block)) {
                    if (targetUses.find(switchCase.targetBlock())->value != 1)
                        continue;

                    addOverride(
                        block, branch->child(0),
                        Override::constant(switchCase.targetBlock(), switchCase.caseValue()));
                }
                break;
            }
            default:
                break;
            }
        }

        // Install the constants in the override blocks. We use one-shot insertion sets because
        // each block will get at most one thing inserted into it anyway.
        for (auto& entry : overrides) {
            for (Override& override : entry.value) {
                if (!override.hasValue)
                    continue;
                override.valueNode =
                    m_insertionSet.insertIntConstant(0, entry.key, override.value);
                m_insertionSet.execute(override.block);
            }
        }

        // Replace all uses of a value that has an override with that override, if appropriate.
        // Certain instructions get special treatment.
        auto getOverride = [&] (BasicBlock* block, Value* value) -> Override {
            auto iter = overrides.find(value);
            if (iter == overrides.end())
                return Override();

            Vector<Override>& forValue = iter->value;
            Override result;
            for (Override& override : forValue) {
                if (dominators.dominates(override.block, block)
                    && override.isBetterThan(result))
                    result = override;
            }

            if (B3FoldPathConstantsInternal::verbose)
                dataLog("In block ", *block, " getting override for ", *value, ": ", result, "\n");

            return result;
        };
        
        for (BasicBlock* block : m_proc) {
            for (unsigned valueIndex = 0; valueIndex < block->size(); ++valueIndex) {
                Value* value = block->at(valueIndex);

                switch (value->opcode()) {
                case Branch: {
                    if (getOverride(block, value->child(0)).isNonZero) {
                        value->replaceWithJump(block, block->taken());
                        changed = true;
                    }
                    break;
                }

                case Equal: {
                    if (value->child(1)->isInt(0)
                        && getOverride(block, value->child(0)).isNonZero) {
                        value->replaceWithIdentity(
                            m_insertionSet.insertIntConstant(valueIndex, value, 0));
                    }
                    break;
                }

                case NotEqual: {
                    if (value->child(1)->isInt(0)
                        && getOverride(block, value->child(0)).isNonZero) {
                        value->replaceWithIdentity(
                            m_insertionSet.insertIntConstant(valueIndex, value, 1));
                    }
                    break;
                }

                default:
                    break;
                }

                for (Value*& child : value->children()) {
                    Override override = getOverride(block, child);
                    if (override.valueNode)
                        child = override.valueNode;
                }
            }
            m_insertionSet.execute(block);
        }

        if (changed) {
            m_proc.resetReachability();
            m_proc.invalidateCFG();
        }
    }
    
private:
    struct Override {
        Override()
        {
        }

        static Override constant(BasicBlock* block, int64_t value)
        {
            Override result;
            result.block = block;
            result.hasValue = true;
            result.value = value;
            if (value)
                result.isNonZero = true;
            return result;
        }

        static Override nonZero(BasicBlock* block)
        {
            Override result;
            result.block = block;
            result.isNonZero = true;
            return result;
        }

        bool isBetterThan(const Override& override)
        {
            if (hasValue && !override.hasValue)
                return true;
            if (isNonZero && !override.isNonZero)
                return true;
            return false;
        }

        void dump(PrintStream& out) const
        {
            out.print("{block = ", pointerDump(block), ", value = ");
            if (hasValue)
                out.print(value);
            else
                out.print("<none>");
            out.print(", isNonZero = ", isNonZero);
            if (valueNode)
                out.print(", valueNode = ", *valueNode);
            out.print("}");
        }

        BasicBlock* block { nullptr };
        bool hasValue { false };
        bool isNonZero { false };
        int64_t value { 0 };
        Value* valueNode { nullptr };
    };

    Procedure& m_proc;
    InsertionSet m_insertionSet;
};

} // anonymous namespace

void foldPathConstants(Procedure& proc)
{
    PhaseScope phaseScope(proc, "foldPathConstants"_s);
    FoldPathConstants foldPathConstants(proc);
    foldPathConstants.run();
}

} } // namespace JSC::B3

#endif // ENABLE(B3_JIT)

