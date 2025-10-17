/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 14, 2023.
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
#include <wtf/FixedVector.h>

namespace JSC {

class CachedEvalCodeBlock;

class UnlinkedEvalCodeBlock final : public UnlinkedGlobalCodeBlock {
public:
    typedef UnlinkedGlobalCodeBlock Base;
    static constexpr unsigned StructureFlags = Base::StructureFlags | StructureIsImmortal;

    template<typename CellType, SubspaceAccess mode>
    static GCClient::IsoSubspace* subspaceFor(VM& vm)
    {
        return vm.unlinkedEvalCodeBlockSpace<mode>();
    }

    static UnlinkedEvalCodeBlock* create(VM& vm, const ExecutableInfo& info, OptionSet<CodeGenerationMode> codeGenerationMode)
    {
        UnlinkedEvalCodeBlock* instance = new (NotNull, allocateCell<UnlinkedEvalCodeBlock>(vm)) UnlinkedEvalCodeBlock(vm, vm.unlinkedEvalCodeBlockStructure.get(), info, codeGenerationMode);
        instance->finishCreation(vm);
        return instance;
    }

    static void destroy(JSCell*);

    const Identifier& variable(unsigned index) { return m_variables[index]; }
    unsigned numVariables() { return m_variables.size(); }
    void adoptVariables(Vector<Identifier, 0, UnsafeVectorOverflow>&& variables)
    {
        ASSERT(m_variables.isEmpty());
        m_variables = FixedVector<Identifier>(WTFMove(variables));
    }

    const Identifier& functionHoistingCandidate(unsigned index) { return m_functionHoistingCandidates[index]; }
    unsigned numFunctionHoistingCandidates() { return m_functionHoistingCandidates.size(); }
    void adoptFunctionHoistingCandidates(Vector<Identifier, 0, UnsafeVectorOverflow>&& functionHoistingCandidates)
    {
        ASSERT(m_functionHoistingCandidates.isEmpty());
        m_functionHoistingCandidates = FixedVector<Identifier>(WTFMove(functionHoistingCandidates));
    }
private:
    friend CachedEvalCodeBlock;

    UnlinkedEvalCodeBlock(VM& vm, Structure* structure, const ExecutableInfo& info, OptionSet<CodeGenerationMode> codeGenerationMode)
        : Base(vm, structure, EvalCode, info, codeGenerationMode)
    {
    }

    UnlinkedEvalCodeBlock(Decoder&, const CachedEvalCodeBlock&);

    FixedVector<Identifier> m_variables;
    FixedVector<Identifier> m_functionHoistingCandidates;

public:
    inline static Structure* createStructure(VM&, JSGlobalObject*, JSValue);

    DECLARE_INFO;
};

}
