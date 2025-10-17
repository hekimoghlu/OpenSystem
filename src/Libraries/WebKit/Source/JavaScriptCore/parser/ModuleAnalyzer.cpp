/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 2, 2024.
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
#include "ModuleAnalyzer.h"

#include "IdentifierInlines.h"
#include "JSGlobalObject.h"
#include "JSModuleRecord.h"
#include "ModuleScopeData.h"
#include "StrongInlines.h"

namespace JSC {

ModuleAnalyzer::ModuleAnalyzer(JSGlobalObject* globalObject, const Identifier& moduleKey, const SourceCode& sourceCode, const VariableEnvironment& declaredVariables, const VariableEnvironment& lexicalVariables, CodeFeatures features)
    : m_vm(globalObject->vm())
    , m_moduleRecord(JSModuleRecord::create(globalObject, m_vm, globalObject->moduleRecordStructure(), moduleKey, sourceCode, declaredVariables, lexicalVariables, features))
{
}

void ModuleAnalyzer::appendRequestedModule(const Identifier& specifier, RefPtr<ScriptFetchParameters>&& attributes)
{
    auto result = m_requestedModules.add(specifier.impl());
    if (result.isNewEntry)
        moduleRecord()->appendRequestedModule(specifier, WTFMove(attributes));
}

void ModuleAnalyzer::exportVariable(ModuleProgramNode& moduleProgramNode, const RefPtr<UniquedStringImpl>& localName, const VariableEnvironmentEntry& variable)
{
    // In the parser, we already marked the variables as Exported and Imported.
    // By leveraging this information, we collect the information that is needed
    // to construct the module environment.
    //
    // I E
    //   * = exported module local variable
    // *   = imported binding
    //     = non-exported module local variable
    // * * = indirect exported binding
    //
    // One exception is namespace binding (like import * as ns from "mod").
    // This is annotated as an imported, but the actual binding is locate in the
    // current module.

    if (!variable.isExported())
        return;

    // Exported module local variable.
    if (!variable.isImported()) {
        for (auto& exportName : moduleProgramNode.moduleScopeData().exportedBindings().get(localName.get()))
            moduleRecord()->addExportEntry(JSModuleRecord::ExportEntry::createLocal(Identifier::fromUid(m_vm, exportName.get()), Identifier::fromUid(m_vm, localName.get())));
        return;
    }

    if (variable.isImportedNamespace()) {
        // Exported namespace binding.
        // import * as namespace from "mod"
        // export { namespace }
        //
        // Sec 15.2.1.16.1 step 11-a-ii-2-b https://tc39.github.io/ecma262/#sec-parsemodule
        // Namespace export is handled as local export since a namespace object binding itself is implemented as a local binding.
        for (auto& exportName : moduleProgramNode.moduleScopeData().exportedBindings().get(localName.get()))
            moduleRecord()->addExportEntry(JSModuleRecord::ExportEntry::createLocal(Identifier::fromUid(m_vm, exportName.get()), Identifier::fromUid(m_vm, localName.get())));
        return;
    }

    // Indirectly exported binding.
    // import a from "mod"
    // export { a }
    std::optional<JSModuleRecord::ImportEntry> optionalImportEntry = moduleRecord()->tryGetImportEntry(localName.get());
    ASSERT(optionalImportEntry);
    const JSModuleRecord::ImportEntry& importEntry = *optionalImportEntry;
    for (auto& exportName : moduleProgramNode.moduleScopeData().exportedBindings().get(localName.get()))
        moduleRecord()->addExportEntry(JSModuleRecord::ExportEntry::createIndirect(Identifier::fromUid(m_vm, exportName.get()), importEntry.importName, importEntry.moduleRequest));
}



Expected<JSModuleRecord*, std::tuple<ErrorType, String>> ModuleAnalyzer::analyze(ModuleProgramNode& moduleProgramNode)
{
    // Traverse the module AST and collect
    // * Import entries
    // * Export entries that have FromClause (e.g. export { a } from "mod")
    // * Export entries that have star (e.g. export * from "mod")
    // * Aliased export names (e.g. export { a as b })
    if (!moduleProgramNode.analyzeModule(*this))
        return makeUnexpected(WTFMove(m_errorMessage));

    // Based on the collected information, categorize export entries into 3 types.
    // 1. Local export entries
    //     This references the local variable in the current module.
    //     This variable should be allocated in the current module environment as a heap variable.
    //
    //     const variable = 20
    //     export { variable }
    //
    // 2. Namespace export entries
    //     This references the namespace object imported by some import entries.
    //     This variable itself should be allocated in the current module environment as a heap variable.
    //     But when the other modules attempt to resolve this export name in this module, this module
    //     should tell the link to the original module.
    //
    //     import * as namespace from "mod"
    //     export { namespace as mod }
    //
    // 3. Indirect export entries
    //     This references the imported binding name from the other module.
    //     This module environment itself should hold the pointer to (1) the original module and
    //     (2) the binding in the original module. The variable itself is allocated in the original
    //     module. This indirect binding is resolved when the CodeBlock resolves the references.
    //
    //     import mod from "mod"
    //     export { mod }
    //
    //     export { a } from "mod"
    //
    // And separeted from the above 3 types, we also collect the star export entries.
    //
    // 4. Star export entries
    //     This exports all the names from the specified external module as the current module's name.
    //
    //     export * from "mod"
    for (const auto& pair : m_moduleRecord->declaredVariables())
        exportVariable(moduleProgramNode, pair.key, pair.value);

    for (const auto& pair : m_moduleRecord->lexicalVariables())
        exportVariable(moduleProgramNode, pair.key, pair.value);

    if (UNLIKELY(Options::dumpModuleRecord()))
        m_moduleRecord->dump();

    return m_moduleRecord;
}

} // namespace JSC
