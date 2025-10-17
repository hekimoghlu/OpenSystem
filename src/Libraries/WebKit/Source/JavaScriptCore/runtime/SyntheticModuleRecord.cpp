/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 16, 2023.
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
#include "SyntheticModuleRecord.h"

#include "BuiltinNames.h"
#include "JSCInlines.h"
#include "JSInternalPromise.h"
#include "JSModuleEnvironment.h"
#include "JSModuleNamespaceObject.h"
#include "JSONObject.h"

namespace JSC {

const ClassInfo SyntheticModuleRecord::s_info = { "ModuleRecord"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(SyntheticModuleRecord) };


Structure* SyntheticModuleRecord::createStructure(VM& vm, JSGlobalObject* globalObject, JSValue prototype)
{
    return Structure::create(vm, globalObject, prototype, TypeInfo(ObjectType, StructureFlags), info());
}

SyntheticModuleRecord* SyntheticModuleRecord::create(JSGlobalObject* globalObject, VM& vm, Structure* structure, const Identifier& moduleKey)
{
    SyntheticModuleRecord* instance = new (NotNull, allocateCell<SyntheticModuleRecord>(vm)) SyntheticModuleRecord(vm, structure, moduleKey);
    instance->finishCreation(globalObject, vm);
    return instance;
}

SyntheticModuleRecord::SyntheticModuleRecord(VM& vm, Structure* structure, const Identifier& moduleKey)
    : Base(vm, structure, moduleKey)
{
}

void SyntheticModuleRecord::destroy(JSCell* cell)
{
    SyntheticModuleRecord* thisObject = static_cast<SyntheticModuleRecord*>(cell);
    thisObject->SyntheticModuleRecord::~SyntheticModuleRecord();
}

void SyntheticModuleRecord::finishCreation(JSGlobalObject* globalObject, VM& vm)
{
    Base::finishCreation(globalObject, vm);
    ASSERT(inherits(info()));
}

template<typename Visitor>
void SyntheticModuleRecord::visitChildrenImpl(JSCell* cell, Visitor& visitor)
{
    SyntheticModuleRecord* thisObject = jsCast<SyntheticModuleRecord*>(cell);
    ASSERT_GC_OBJECT_INHERITS(thisObject, info());
    Base::visitChildren(thisObject, visitor);
}

DEFINE_VISIT_CHILDREN(SyntheticModuleRecord);

Synchronousness SyntheticModuleRecord::link(JSGlobalObject*, JSValue)
{
    return Synchronousness::Sync;
}

JSValue SyntheticModuleRecord::evaluate(JSGlobalObject*)
{
    return jsUndefined();
}


SyntheticModuleRecord* SyntheticModuleRecord::tryCreateWithExportNamesAndValues(JSGlobalObject* globalObject, const Identifier& moduleKey, const Vector<Identifier, 4>& exportNames, const MarkedArgumentBuffer& exportValues)
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    ASSERT(exportNames.size() == exportValues.size());

    auto* moduleRecord = create(globalObject, vm, globalObject->syntheticModuleRecordStructure(), moduleKey);

    SymbolTable* exportSymbolTable = SymbolTable::create(vm);
    {
        auto offset = exportSymbolTable->takeNextScopeOffset(NoLockingNecessary);
        exportSymbolTable->set(NoLockingNecessary, vm.propertyNames->starNamespacePrivateName.impl(), SymbolTableEntry(VarOffset(offset)));
    }
    for (auto& exportName : exportNames) {
        auto offset = exportSymbolTable->takeNextScopeOffset(NoLockingNecessary);
        exportSymbolTable->set(NoLockingNecessary, exportName.impl(), SymbolTableEntry(VarOffset(offset)));
        moduleRecord->addExportEntry(ExportEntry::createLocal(exportName, exportName));
    }

    JSModuleEnvironment* moduleEnvironment = JSModuleEnvironment::create(vm, globalObject, nullptr, exportSymbolTable, jsTDZValue(), moduleRecord);
    moduleRecord->setModuleEnvironment(globalObject, moduleEnvironment);
    RETURN_IF_EXCEPTION(scope, { });

    for (unsigned index = 0; index < exportNames.size(); ++index) {
        PropertyName exportName = exportNames[index];
        JSValue exportValue = exportValues.at(index);
        constexpr bool shouldThrowReadOnlyError = false;
        constexpr bool ignoreReadOnlyErrors = true;
        bool putResult = false;
        symbolTablePutTouchWatchpointSet(moduleEnvironment, globalObject, exportName, exportValue, shouldThrowReadOnlyError, ignoreReadOnlyErrors, putResult);
        RETURN_IF_EXCEPTION(scope, { });
        ASSERT(putResult);
    }

    return moduleRecord;

}

SyntheticModuleRecord* SyntheticModuleRecord::tryCreateDefaultExportSyntheticModule(JSGlobalObject* globalObject, const Identifier& moduleKey, JSValue defaultExport)
{
    VM& vm = globalObject->vm();

    Vector<Identifier, 4> exportNames;
    MarkedArgumentBuffer exportValues;

    exportNames.append(vm.propertyNames->defaultKeyword);
    exportValues.appendWithCrashOnOverflow(defaultExport);

    return tryCreateWithExportNamesAndValues(globalObject, moduleKey, exportNames, exportValues);
}

SyntheticModuleRecord* SyntheticModuleRecord::parseJSONModule(JSGlobalObject* globalObject, const Identifier& moduleKey, SourceCode&& sourceCode)
{
    // https://tc39.es/proposal-json-modules/#sec-parse-json-module
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    JSValue result = JSONParseWithException(globalObject, sourceCode.view());
    RETURN_IF_EXCEPTION(scope, { });

    RELEASE_AND_RETURN(scope, SyntheticModuleRecord::tryCreateDefaultExportSyntheticModule(globalObject, moduleKey, result));
}

} // namespace JSC
