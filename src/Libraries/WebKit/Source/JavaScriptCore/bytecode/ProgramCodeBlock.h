/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 4, 2022.
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

#include "GlobalCodeBlock.h"
#include "UnlinkedProgramCodeBlock.h"

namespace JSC {

class ProgramCodeBlock final : public GlobalCodeBlock {
public:
    typedef GlobalCodeBlock Base;
    DECLARE_INFO;

    template<typename, SubspaceAccess>
    static GCClient::IsoSubspace* subspaceFor(VM& vm)
    {
        return &vm.codeBlockSpace();
    }

    static ProgramCodeBlock* create(VM& vm, CopyParsedBlockTag, ProgramCodeBlock& other)
    {
        ProgramCodeBlock* instance = new (NotNull, allocateCell<ProgramCodeBlock>(vm))
            ProgramCodeBlock(vm, vm.programCodeBlockStructure.get(), CopyParsedBlock, other);
        instance->finishCreation(vm, CopyParsedBlock, other);
        return instance;
    }

    static ProgramCodeBlock* create(VM& vm, ProgramExecutable* ownerExecutable, UnlinkedProgramCodeBlock* unlinkedCodeBlock, JSScope* scope)
    {
        ProgramCodeBlock* instance = new (NotNull, allocateCell<ProgramCodeBlock>(vm))
            ProgramCodeBlock(vm, vm.programCodeBlockStructure.get(), ownerExecutable, unlinkedCodeBlock, scope);
        if (!instance->finishCreation(vm, ownerExecutable, unlinkedCodeBlock, scope))
            return nullptr;
        return instance;
    }

    inline static Structure* createStructure(VM&, JSGlobalObject*, JSValue);

private:
    ProgramCodeBlock(VM& vm, Structure* structure, CopyParsedBlockTag, ProgramCodeBlock& other)
        : GlobalCodeBlock(vm, structure, CopyParsedBlock, other)
    {
    }

    ProgramCodeBlock(VM& vm, Structure* structure, ProgramExecutable* ownerExecutable, UnlinkedProgramCodeBlock* unlinkedCodeBlock, JSScope* scope)
        : GlobalCodeBlock(vm, structure, ownerExecutable, unlinkedCodeBlock, scope)
    {
    }
};
static_assert(sizeof(ProgramCodeBlock) == sizeof(CodeBlock), "Subclasses of CodeBlock should be the same size to share IsoSubspace");

} // namespace JSC
