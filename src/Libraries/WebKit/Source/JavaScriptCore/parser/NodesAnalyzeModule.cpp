/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 11, 2022.
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
#include "Nodes.h"

#include "JSCJSValueInlines.h"
#include "JSModuleRecord.h"
#include "ModuleAnalyzer.h"
#include <wtf/text/MakeString.h>

namespace JSC {

static Expected<RefPtr<ScriptFetchParameters>, std::tuple<ErrorType, String>> tryCreateAttributes(VM& vm, ImportAttributesListNode* attributesList)
{
    if (!attributesList)
        return RefPtr<ScriptFetchParameters> { };

    // https://tc39.es/proposal-import-attributes/#sec-AllImportAttributesSupported
    // Currently, only "type" is supported.
    std::optional<ScriptFetchParameters::Type> type;
    for (auto& [key, value] : attributesList->attributes()) {
        if (*key != vm.propertyNames->type)
            return makeUnexpected(std::tuple { ErrorType::SyntaxError, makeString("Import attribute \""_s, StringView(key->impl()), "\" is not supported"_s) });
    }

    for (auto& [key, value] : attributesList->attributes()) {
        if (*key == vm.propertyNames->type) {
            type = ScriptFetchParameters::parseType(value->impl());
            if (!type)
                return makeUnexpected(std::tuple { ErrorType::TypeError, makeString("Import attribute type \""_s, StringView(value->impl()), "\" is not valid"_s) });
        }
    }

    if (type)
        return RefPtr<ScriptFetchParameters>(ScriptFetchParameters::create(type.value()));
    return RefPtr<ScriptFetchParameters> { };
}

bool ScopeNode::analyzeModule(ModuleAnalyzer& analyzer)
{
    return m_statements->analyzeModule(analyzer);
}

bool SourceElements::analyzeModule(ModuleAnalyzer& analyzer)
{
    // In the module analyzer phase, only module declarations are included in the top-level SourceElements.
    for (StatementNode* statement = m_head; statement; statement = statement->next()) {
        ASSERT(statement->isModuleDeclarationNode());
        if (!static_cast<ModuleDeclarationNode*>(statement)->analyzeModule(analyzer))
            return false;
    }
    return true;
}

bool ImportDeclarationNode::analyzeModule(ModuleAnalyzer& analyzer)
{
    auto result = tryCreateAttributes(analyzer.vm(), attributesList());
    if (!result) {
        analyzer.fail(WTFMove(result.error()));
        return false;
    }

    analyzer.appendRequestedModule(m_moduleName->moduleName(), WTFMove(result.value()));
    for (auto* specifier : m_specifierList->specifiers()) {
        analyzer.moduleRecord()->addImportEntry(JSModuleRecord::ImportEntry {
            specifier->importedName() == analyzer.vm().propertyNames->timesIdentifier
                ? JSModuleRecord::ImportEntryType::Namespace : JSModuleRecord::ImportEntryType::Single,
            m_moduleName->moduleName(),
            specifier->importedName(),
            specifier->localName(),
        });
    }
    return true;
}

bool ExportAllDeclarationNode::analyzeModule(ModuleAnalyzer& analyzer)
{
    auto result = tryCreateAttributes(analyzer.vm(), attributesList());
    if (!result) {
        analyzer.fail(WTFMove(result.error()));
        return false;
    }

    analyzer.appendRequestedModule(m_moduleName->moduleName(), WTFMove(result.value()));
    analyzer.moduleRecord()->addStarExportEntry(m_moduleName->moduleName());
    return true;
}

bool ExportDefaultDeclarationNode::analyzeModule(ModuleAnalyzer&)
{
    return true;
}

bool ExportLocalDeclarationNode::analyzeModule(ModuleAnalyzer&)
{
    return true;
}

bool ExportNamedDeclarationNode::analyzeModule(ModuleAnalyzer& analyzer)
{
    if (m_moduleName) {
        auto result = tryCreateAttributes(analyzer.vm(), attributesList());
        if (!result) {
            analyzer.fail(WTFMove(result.error()));
            return false;
        }

        analyzer.appendRequestedModule(m_moduleName->moduleName(), WTFMove(result.value()));
    }

    for (auto* specifier : m_specifierList->specifiers()) {
        if (m_moduleName) {
            // export { v } from "mod"
            //
            // In this case, no local variable names are imported into the current module.
            // "v" indirectly points the binding in "mod".
            //
            // export * as v from "mod"
            //
            // If it is namespace export, we should use createNamespace.
            if (specifier->localName() == analyzer.vm().propertyNames->starNamespacePrivateName)
                analyzer.moduleRecord()->addExportEntry(JSModuleRecord::ExportEntry::createNamespace(specifier->exportedName(), m_moduleName->moduleName()));
            else
                analyzer.moduleRecord()->addExportEntry(JSModuleRecord::ExportEntry::createIndirect(specifier->exportedName(), specifier->localName(), m_moduleName->moduleName()));
        }
    }
    return true;
}

} // namespace JSC
