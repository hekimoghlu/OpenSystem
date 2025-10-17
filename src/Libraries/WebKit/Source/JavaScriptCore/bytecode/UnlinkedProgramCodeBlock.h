/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 25, 2021.
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

#include "UnlinkedGlobalCodeBlock.h"

namespace JSC {

class CachedProgramCodeBlock;

class UnlinkedProgramCodeBlock final : public UnlinkedGlobalCodeBlock {
public:
    typedef UnlinkedGlobalCodeBlock Base;
    static constexpr unsigned StructureFlags = Base::StructureFlags | StructureIsImmortal;

    template<typename CellType, SubspaceAccess mode>
    static GCClient::IsoSubspace* subspaceFor(VM& vm)
    {
        return vm.unlinkedProgramCodeBlockSpace<mode>();
    }

    static UnlinkedProgramCodeBlock* create(VM& vm, const ExecutableInfo& info, OptionSet<CodeGenerationMode> codeGenerationMode)
    {
        UnlinkedProgramCodeBlock* instance = new (NotNull, allocateCell<UnlinkedProgramCodeBlock>(vm)) UnlinkedProgramCodeBlock(vm, vm.unlinkedProgramCodeBlockStructure.get(), info, codeGenerationMode);
        instance->finishCreation(vm);
        return instance;
    }

    static void destroy(JSCell*);

    void setVariableDeclarations(const VariableEnvironment& environment) { m_varDeclarations = environment; }
    const VariableEnvironment& variableDeclarations() const { return m_varDeclarations; }

    void setLexicalDeclarations(const VariableEnvironment& environment) { m_lexicalDeclarations = environment; }
    const VariableEnvironment& lexicalDeclarations() const { return m_lexicalDeclarations; }

private:
    friend CachedProgramCodeBlock;

    UnlinkedProgramCodeBlock(VM& vm, Structure* structure, const ExecutableInfo& info, OptionSet<CodeGenerationMode> codeGenerationMode)
        : Base(vm, structure, GlobalCode, info, codeGenerationMode)
    {
    }

    UnlinkedProgramCodeBlock(Decoder&, const CachedProgramCodeBlock&);

    VariableEnvironment m_varDeclarations;
    VariableEnvironment m_lexicalDeclarations;

public:
    inline static Structure* createStructure(VM&, JSGlobalObject*, JSValue);

    DECLARE_INFO;
};

}
