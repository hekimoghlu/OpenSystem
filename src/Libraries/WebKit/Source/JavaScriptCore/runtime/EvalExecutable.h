/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 2, 2023.
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

#include "GlobalExecutable.h"
#include "UnlinkedEvalCodeBlock.h"

namespace JSC {

class EvalExecutable : public GlobalExecutable {
    friend class LLIntOffsetsExtractor;
public:
    using Base = GlobalExecutable;
    static constexpr unsigned StructureFlags = Base::StructureFlags | StructureIsImmortal;

    static void destroy(JSCell*);
    
    EvalCodeBlock* codeBlock() const
    {
        return std::bit_cast<EvalCodeBlock*>(Base::codeBlock());
    }

    UnlinkedEvalCodeBlock* unlinkedCodeBlock() const
    {
        return std::bit_cast<UnlinkedEvalCodeBlock*>(Base::unlinkedCodeBlock());
    }

    Ref<JSC::JITCode> generatedJITCode()
    {
        return generatedJITCodeForCall();
    }

    inline static Structure* createStructure(VM&, JSGlobalObject*, JSValue);

    template<typename CellType, SubspaceAccess mode>
    static GCClient::IsoSubspace* subspaceFor(VM& vm)
    {
        return vm.evalExecutableSpace<mode>();
    }

    DECLARE_INFO;

    unsigned numVariables() { return unlinkedCodeBlock()->numVariables(); }
    unsigned numFunctionHoistingCandidates() { return unlinkedCodeBlock()->numFunctionHoistingCandidates(); }
    unsigned numTopLevelFunctionDecls() { return unlinkedCodeBlock()->numberOfFunctionDecls(); }
    bool allowDirectEvalCache() const { return unlinkedCodeBlock()->allowDirectEvalCache(); }
    NeedsClassFieldInitializer needsClassFieldInitializer() const { return static_cast<NeedsClassFieldInitializer>(m_needsClassFieldInitializer); }
    PrivateBrandRequirement privateBrandRequirement() const { return static_cast<PrivateBrandRequirement>(m_privateBrandRequirement); }
    TemplateObjectMap& ensureTemplateObjectMap(VM&);

protected:
    friend class ExecutableBase;
    friend class ScriptExecutable;

    using Base::finishCreation;
    EvalExecutable(JSGlobalObject*, const SourceCode&, LexicallyScopedFeatures, DerivedContextType, bool isArrowFunctionContext, bool isInsideOrdinaryFunction, EvalContextType, NeedsClassFieldInitializer, PrivateBrandRequirement);

    DECLARE_VISIT_CHILDREN;

    unsigned m_needsClassFieldInitializer : 1;
    unsigned m_privateBrandRequirement : 1;

    std::unique_ptr<TemplateObjectMap> m_templateObjectMap;
};

} // namespace JSC
