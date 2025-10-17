/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 22, 2021.
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
#include "LLIntPrototypeLoadAdaptiveStructureWatchpoint.h"

#include "CodeBlockInlines.h"
#include "Instruction.h"
#include "JSCellInlines.h"

namespace JSC {

LLIntPrototypeLoadAdaptiveStructureWatchpoint::LLIntPrototypeLoadAdaptiveStructureWatchpoint(CodeBlock* owner, const ObjectPropertyCondition& key, BytecodeIndex bytecodeIndex)
    : Watchpoint(Watchpoint::Type::LLIntPrototypeLoadAdaptiveStructure)
    , m_owner(owner)
    , m_bytecodeIndex(bytecodeIndex)
    , m_key(key)
{
    RELEASE_ASSERT(key.watchingRequiresStructureTransitionWatchpoint());
    RELEASE_ASSERT(!key.watchingRequiresReplacementWatchpoint());
}

LLIntPrototypeLoadAdaptiveStructureWatchpoint::LLIntPrototypeLoadAdaptiveStructureWatchpoint()
    : Watchpoint(Watchpoint::Type::LLIntPrototypeLoadAdaptiveStructure)
    , m_owner(nullptr)
{
}

LLIntPrototypeLoadAdaptiveStructureWatchpoint::~LLIntPrototypeLoadAdaptiveStructureWatchpoint()
{
    ASSERT(!m_owner || !m_owner->wasDestructed());
}

void LLIntPrototypeLoadAdaptiveStructureWatchpoint::initialize(CodeBlock* codeBlock, const ObjectPropertyCondition& key, BytecodeIndex bytecodeOffset)
{
    m_owner = codeBlock;
    m_bytecodeIndex = bytecodeOffset;
    m_key = key;
}

void LLIntPrototypeLoadAdaptiveStructureWatchpoint::install(VM&)
{
    RELEASE_ASSERT(m_key.isWatchable(PropertyCondition::MakeNoChanges));

    m_key.object()->structure()->addTransitionWatchpoint(this);
}

void LLIntPrototypeLoadAdaptiveStructureWatchpoint::fireInternal(VM& vm, const FireDetail&)
{
    ASSERT(!m_owner->wasDestructed());
    if (m_owner->isPendingDestruction())
        return;

    if (m_key.isWatchable(PropertyCondition::EnsureWatchability)) {
        install(vm);
        return;
    }

    auto& instruction = m_owner->instructions().at(m_bytecodeIndex.get().offset());
    switch (instruction->opcodeID()) {
    case op_get_by_id:
        clearLLIntGetByIdCache(instruction->as<OpGetById>().metadata(m_owner.get()).m_modeMetadata);
        break;

    case op_get_length:
        clearLLIntGetByIdCache(instruction->as<OpGetLength>().metadata(m_owner.get()).m_modeMetadata);
        break;

    case op_iterator_open:
        clearLLIntGetByIdCache(instruction->as<OpIteratorOpen>().metadata(m_owner.get()).m_modeMetadata);
        break;

    case op_iterator_next: {
        auto& metadata = instruction->as<OpIteratorNext>().metadata(m_owner.get());
        switch (m_bytecodeIndex.get().checkpoint()) {
        case OpIteratorNext::getDone:
            clearLLIntGetByIdCache(metadata.m_doneModeMetadata);
            break;
        case OpIteratorNext::getValue:
            clearLLIntGetByIdCache(metadata.m_valueModeMetadata);
            break;
        default:
            RELEASE_ASSERT_NOT_REACHED();
        }
        break;
    }

    case op_instanceof: {
        auto& metadata = instruction->as<OpInstanceof>().metadata(m_owner.get());
        switch (m_bytecodeIndex.get().checkpoint()) {
        case OpInstanceof::getPrototype:
            clearLLIntGetByIdCache(metadata.m_hasInstanceModeMetadata);
            break;
        case OpInstanceof::instanceof:
            clearLLIntGetByIdCache(metadata.m_prototypeModeMetadata);
            break;
        default:
            RELEASE_ASSERT_NOT_REACHED();
        }
        break;
    }

    default:
        RELEASE_ASSERT_NOT_REACHED();
        break;
    }
}

void LLIntPrototypeLoadAdaptiveStructureWatchpoint::clearLLIntGetByIdCache(GetByIdModeMetadata& metadata)
{
    metadata.clearToDefaultModeWithoutCache();
}

} // namespace JSC
