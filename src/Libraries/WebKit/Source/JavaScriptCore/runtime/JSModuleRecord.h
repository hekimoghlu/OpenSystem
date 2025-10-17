/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 15, 2023.
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

#include "AbstractModuleRecord.h"
#include "SourceCode.h"
#include "VariableEnvironment.h"

namespace JSC {

class ModuleProgramExecutable;

// Based on the Source Text Module Record
// http://www.ecma-international.org/ecma-262/6.0/#sec-source-text-module-records
class JSModuleRecord final : public AbstractModuleRecord {
    friend class LLIntOffsetsExtractor;
public:
    using Base = AbstractModuleRecord;

    DECLARE_EXPORT_INFO;

    static constexpr DestructionMode needsDestruction = NeedsDestruction;
    static void destroy(JSCell*);

    template<typename CellType, SubspaceAccess mode>
    static GCClient::IsoSubspace* subspaceFor(VM& vm)
    {
        return vm.jsModuleRecordSpace<mode>();
    }

    static Structure* createStructure(VM&, JSGlobalObject*, JSValue);
    static JSModuleRecord* create(JSGlobalObject*, VM&, Structure*, const Identifier&, const SourceCode&, const VariableEnvironment&, const VariableEnvironment&, CodeFeatures);

    Synchronousness link(JSGlobalObject*, JSValue scriptFetcher);
    JS_EXPORT_PRIVATE JSValue evaluate(JSGlobalObject*, JSValue sentValue, JSValue resumeMode);

    const SourceCode& sourceCode() const { return m_sourceCode; }
    const VariableEnvironment& declaredVariables() const { return m_declaredVariables; }
    const VariableEnvironment& lexicalVariables() const { return m_lexicalVariables; }

private:
    JSModuleRecord(VM&, Structure*, const Identifier&, const SourceCode&, const VariableEnvironment&, const VariableEnvironment&, CodeFeatures);

    void finishCreation(JSGlobalObject*, VM&);

    DECLARE_VISIT_CHILDREN;

    void instantiateDeclarations(JSGlobalObject*, ModuleProgramExecutable*, JSValue scriptFetcher);

    SourceCode m_sourceCode;
    VariableEnvironment m_declaredVariables;
    VariableEnvironment m_lexicalVariables;
    WriteBarrier<ModuleProgramExecutable> m_moduleProgramExecutable;
    CodeFeatures m_features;
};

} // namespace JSC
