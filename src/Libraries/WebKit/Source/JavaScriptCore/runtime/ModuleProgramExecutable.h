/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 13, 2023.
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

namespace JSC {

class UnlinkedModuleProgramCodeBlock;

class ModuleProgramExecutable final : public GlobalExecutable {
    friend class LLIntOffsetsExtractor;
public:
    using Base = GlobalExecutable;
    static constexpr unsigned StructureFlags = Base::StructureFlags | StructureIsImmortal;

    template<typename CellType, SubspaceAccess mode>
    static GCClient::IsoSubspace* subspaceFor(VM& vm)
    {
        return vm.moduleProgramExecutableSpace<mode>();
    }

    static ModuleProgramExecutable* create(JSGlobalObject*, const SourceCode&);

    static void destroy(JSCell*);

    ModuleProgramCodeBlock* codeBlock() const
    {
        return std::bit_cast<ModuleProgramCodeBlock*>(Base::codeBlock());
    }

    UnlinkedModuleProgramCodeBlock* getUnlinkedCodeBlock(JSGlobalObject*);

    UnlinkedModuleProgramCodeBlock* unlinkedCodeBlock() const
    {
        return std::bit_cast<UnlinkedModuleProgramCodeBlock*>(Base::unlinkedCodeBlock());
    }

    Ref<JSC::JITCode> generatedJITCode()
    {
        return generatedJITCodeForCall();
    }

    inline static Structure* createStructure(VM&, JSGlobalObject*, JSValue);

    DECLARE_INFO;

    bool isAsync() const { return features() & AwaitFeature; }

    SymbolTable* moduleEnvironmentSymbolTable() { return m_moduleEnvironmentSymbolTable.get(); }

    TemplateObjectMap& ensureTemplateObjectMap(VM&);

private:
    friend class ExecutableBase;
    friend class ScriptExecutable;

    ModuleProgramExecutable(JSGlobalObject*, const SourceCode&);

    DECLARE_VISIT_CHILDREN;

    WriteBarrier<SymbolTable> m_moduleEnvironmentSymbolTable;
    std::unique_ptr<TemplateObjectMap> m_templateObjectMap;
};

} // namespace JSC
