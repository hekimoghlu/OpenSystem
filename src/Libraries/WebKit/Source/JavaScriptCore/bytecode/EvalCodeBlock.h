/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 18, 2024.
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

namespace JSC {

class EvalCodeBlock final : public GlobalCodeBlock {
public:
    typedef GlobalCodeBlock Base;
    DECLARE_INFO;

    template<typename, SubspaceAccess>
    static GCClient::IsoSubspace* subspaceFor(VM& vm)
    {
        return &vm.codeBlockSpace();
    }

    static EvalCodeBlock* create(VM& vm, CopyParsedBlockTag, EvalCodeBlock& other)
    {
        EvalCodeBlock* instance = new (NotNull, allocateCell<EvalCodeBlock>(vm))
            EvalCodeBlock(vm, vm.evalCodeBlockStructure.get(), CopyParsedBlock, other);
        instance->finishCreation(vm, CopyParsedBlock, other);
        return instance;
    }

    static EvalCodeBlock* create(VM& vm, EvalExecutable* ownerExecutable, UnlinkedEvalCodeBlock* unlinkedCodeBlock, JSScope* scope)
    {
        EvalCodeBlock* instance = new (NotNull, allocateCell<EvalCodeBlock>(vm))
            EvalCodeBlock(vm, vm.evalCodeBlockStructure.get(), ownerExecutable, unlinkedCodeBlock, scope);
        if (!instance->finishCreation(vm, ownerExecutable, unlinkedCodeBlock, scope))
            return nullptr;
        return instance;
    }

    inline static Structure* createStructure(VM&, JSGlobalObject*, JSValue);

    UnlinkedEvalCodeBlock* unlinkedEvalCodeBlock() const { return jsCast<UnlinkedEvalCodeBlock*>(unlinkedCodeBlock()); }

private:
    EvalCodeBlock(VM& vm, Structure* structure, CopyParsedBlockTag, EvalCodeBlock& other)
        : GlobalCodeBlock(vm, structure, CopyParsedBlock, other)
    {
    }
        
    EvalCodeBlock(VM& vm, Structure* structure, EvalExecutable* ownerExecutable, UnlinkedEvalCodeBlock* unlinkedCodeBlock, JSScope* scope)
        : GlobalCodeBlock(vm, structure, ownerExecutable, unlinkedCodeBlock, scope)
    {
    }
};
static_assert(sizeof(EvalCodeBlock) == sizeof(CodeBlock), "Subclasses of CodeBlock should be the same size to share IsoSubspace");

} // namespace JSC
