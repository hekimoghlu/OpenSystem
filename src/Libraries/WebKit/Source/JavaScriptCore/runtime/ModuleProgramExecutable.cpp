/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 19, 2022.
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

#include "CodeCache.h"
#include "Debugger.h"

namespace JSC {

const ClassInfo ModuleProgramExecutable::s_info = { "ModuleProgramExecutable"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(ModuleProgramExecutable) };

ModuleProgramExecutable::ModuleProgramExecutable(JSGlobalObject* globalObject, const SourceCode& source)
    : Base(globalObject->vm().moduleProgramExecutableStructure.get(), globalObject->vm(), source, StrictModeLexicallyScopedFeature, DerivedContextType::None, false, false, EvalContextType::None, NoIntrinsic)
{
    ASSERT(source.provider()->sourceType() == SourceProviderSourceType::Module);
    VM& vm = globalObject->vm();
    if (vm.typeProfiler() || vm.controlFlowProfiler())
        vm.functionHasExecutedCache()->insertUnexecutedRange(sourceID(), typeProfilingStartOffset(), typeProfilingEndOffset());
}


UnlinkedModuleProgramCodeBlock* ModuleProgramExecutable::getUnlinkedCodeBlock(JSGlobalObject* globalObject)
{
    VM& vm = globalObject->vm();
    auto throwScope = DECLARE_THROW_SCOPE(vm);

    UnlinkedModuleProgramCodeBlock* unlinkedModuleProgramCode = unlinkedCodeBlock();
    if (unlinkedModuleProgramCode)
        RELEASE_AND_RETURN(throwScope, unlinkedModuleProgramCode);

    ParserError error;
    OptionSet<CodeGenerationMode> codeGenerationMode = globalObject->defaultCodeGenerationMode();
    unlinkedModuleProgramCode = vm.codeCache()->getUnlinkedModuleProgramCodeBlock(vm, this, source(), codeGenerationMode, error);

    if (globalObject->hasDebugger())
        globalObject->debugger()->sourceParsed(globalObject, source().provider(), error.line(), error.message());

    if (error.isValid()) {
        throwVMError(globalObject, throwScope, error.toErrorObject(globalObject, source()));
        return nullptr;
    }

    m_unlinkedCodeBlock.set(vm, this, unlinkedModuleProgramCode);
    VirtualRegister symbolTableReg = VirtualRegister(unlinkedModuleProgramCode->moduleEnvironmentSymbolTableConstantRegisterOffset());
    SymbolTable* symbolTable = jsCast<SymbolTable*>(unlinkedModuleProgramCode->getConstant(symbolTableReg));
    m_moduleEnvironmentSymbolTable.set(vm, this, symbolTable->cloneScopePart(vm));
    RELEASE_AND_RETURN(throwScope, unlinkedModuleProgramCode);
}

ModuleProgramExecutable* ModuleProgramExecutable::create(JSGlobalObject* globalObject, const SourceCode& source)
{
    VM& vm = globalObject->vm();
    ModuleProgramExecutable* executable = new (NotNull, allocateCell<ModuleProgramExecutable>(vm)) ModuleProgramExecutable(globalObject, source);
    executable->finishCreation(vm);
    executable->getUnlinkedCodeBlock(globalObject); // This generates and binds unlinked code block.
    return executable;
}

void ModuleProgramExecutable::destroy(JSCell* cell)
{
    static_cast<ModuleProgramExecutable*>(cell)->ModuleProgramExecutable::~ModuleProgramExecutable();
}

auto ModuleProgramExecutable::ensureTemplateObjectMap(VM&) -> TemplateObjectMap&
{
    return ensureTemplateObjectMapImpl(m_templateObjectMap);
}

template<typename Visitor>
void ModuleProgramExecutable::visitChildrenImpl(JSCell* cell, Visitor& visitor)
{
    ModuleProgramExecutable* thisObject = jsCast<ModuleProgramExecutable*>(cell);
    ASSERT_GC_OBJECT_INHERITS(thisObject, info());
    Base::visitChildren(thisObject, visitor);
    visitor.append(thisObject->m_moduleEnvironmentSymbolTable);
    if (TemplateObjectMap* map = thisObject->m_templateObjectMap.get()) {
        Locker locker { thisObject->cellLock() };
        for (auto& entry : *map)
            visitor.append(entry.value);
    }
}

DEFINE_VISIT_CHILDREN(ModuleProgramExecutable);

} // namespace JSC
