/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 10, 2025.
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
#include "UnlinkedModuleProgramCodeBlock.h"

namespace JSC {

class ModuleProgramCodeBlock final : public GlobalCodeBlock {
public:
    typedef GlobalCodeBlock Base;
    DECLARE_INFO;

    template<typename, SubspaceAccess>
    static GCClient::IsoSubspace* subspaceFor(VM& vm)
    {
        return &vm.codeBlockSpace();
    }

    static ModuleProgramCodeBlock* create(VM& vm, CopyParsedBlockTag, ModuleProgramCodeBlock& other)
    {
        ModuleProgramCodeBlock* instance = new (NotNull, allocateCell<ModuleProgramCodeBlock>(vm))
            ModuleProgramCodeBlock(vm, vm.moduleProgramCodeBlockStructure.get(), CopyParsedBlock, other);
        instance->finishCreation(vm, CopyParsedBlock, other);
        return instance;
    }

    static ModuleProgramCodeBlock* create(VM& vm, ModuleProgramExecutable* ownerExecutable, UnlinkedModuleProgramCodeBlock* unlinkedCodeBlock, JSScope* scope)
    {
        ModuleProgramCodeBlock* instance = new (NotNull, allocateCell<ModuleProgramCodeBlock>(vm))
            ModuleProgramCodeBlock(vm, vm.moduleProgramCodeBlockStructure.get(), ownerExecutable, unlinkedCodeBlock, scope);
        if (!instance->finishCreation(vm, ownerExecutable, unlinkedCodeBlock, scope))
            return nullptr;
        return instance;
    }

    inline static Structure* createStructure(VM&, JSGlobalObject*, JSValue);

private:
    ModuleProgramCodeBlock(VM& vm, Structure* structure, CopyParsedBlockTag, ModuleProgramCodeBlock& other)
        : GlobalCodeBlock(vm, structure, CopyParsedBlock, other)
    {
    }

    ModuleProgramCodeBlock(VM& vm, Structure* structure, ModuleProgramExecutable* ownerExecutable, UnlinkedModuleProgramCodeBlock* unlinkedCodeBlock, JSScope* scope)
        : GlobalCodeBlock(vm, structure, ownerExecutable, unlinkedCodeBlock, scope)
    {
    }
};
static_assert(sizeof(ModuleProgramCodeBlock) == sizeof(CodeBlock), "Subclasses of CodeBlock should be the same size to share IsoSubspace");

} // namespace JSC
