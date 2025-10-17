/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 5, 2024.
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
#include "JSModuleEnvironment.h"

#include "AbstractModuleRecord.h"
#include "JSCInlines.h"

namespace JSC {

const ClassInfo JSModuleEnvironment::s_info = { "JSModuleEnvironment"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(JSModuleEnvironment) };

JSModuleEnvironment* JSModuleEnvironment::create(
    VM& vm, Structure* structure, JSScope* currentScope, SymbolTable* symbolTable, JSValue initialValue, AbstractModuleRecord* moduleRecord)
{
    // JSLexicalEnvironment has the storage to store the variable slots after the its class storage.
    // Because the offset of the variable slots are fixed in the JSLexicalEnvironment, inheritting these class and adding new member field is not allowed,
    // the new member will overlap the variable slots.
    // To keep the JSModuleEnvironment compatible to the JSLexicalEnvironment but add the new member to store the AbstractModuleRecord, we additionally allocate
    // the storage after the variable slots.
    //
    // JSLexicalEnvironment:
    //     [ JSLexicalEnvironment ][ variable slots ]
    //
    // JSModuleEnvironment:
    //     [ JSLexicalEnvironment ][ variable slots ][ additional slots for JSModuleEnvironment ]
    JSModuleEnvironment* result =
        new (
            NotNull,
            allocateCell<JSModuleEnvironment>(vm, JSModuleEnvironment::allocationSize(symbolTable)))
        JSModuleEnvironment(vm, structure, currentScope, symbolTable, initialValue, moduleRecord);
    result->finishCreation(vm);
    return result;
}

template<typename Visitor>
void JSModuleEnvironment::visitChildrenImpl(JSCell* cell, Visitor& visitor)
{
    JSModuleEnvironment* thisObject = jsCast<JSModuleEnvironment*>(cell);
    ASSERT_GC_OBJECT_INHERITS(thisObject, info());
    Base::visitChildren(thisObject, visitor);
    visitor.appendValues(thisObject->variables(), thisObject->symbolTable()->scopeSize());
    visitor.append(thisObject->moduleRecordSlot());
}

DEFINE_VISIT_CHILDREN(JSModuleEnvironment);

bool JSModuleEnvironment::getOwnPropertySlot(JSObject* cell, JSGlobalObject* globalObject, PropertyName propertyName, PropertySlot& slot)
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);
    JSModuleEnvironment* thisObject = jsCast<JSModuleEnvironment*>(cell);
    AbstractModuleRecord::Resolution resolution = thisObject->moduleRecord()->resolveImport(globalObject, Identifier::fromUid(vm, propertyName.uid()));
    RETURN_IF_EXCEPTION(scope, false);
    if (resolution.type == AbstractModuleRecord::Resolution::Type::Resolved) {
        // When resolveImport resolves the resolution, the imported module environment must have the binding.
        JSModuleEnvironment* importedModuleEnvironment = resolution.moduleRecord->moduleEnvironment();
        PropertySlot redirectSlot(importedModuleEnvironment, PropertySlot::InternalMethodType::Get);
        bool result = importedModuleEnvironment->methodTable()->getOwnPropertySlot(importedModuleEnvironment, globalObject, resolution.localName, redirectSlot);
        ASSERT_UNUSED(result, result);
        ASSERT(redirectSlot.isValue());
        JSValue value = redirectSlot.getValue(globalObject, resolution.localName);
        scope.assertNoException();
        slot.setValue(thisObject, redirectSlot.attributes(), value);
        return true;
    }
    return Base::getOwnPropertySlot(thisObject, globalObject, propertyName, slot);
}

void JSModuleEnvironment::getOwnSpecialPropertyNames(JSObject* cell, JSGlobalObject*, PropertyNameArray& propertyNamesArray, DontEnumPropertiesMode)
{
    JSModuleEnvironment* thisObject = jsCast<JSModuleEnvironment*>(cell);
    if (propertyNamesArray.includeStringProperties()) {
        for (const auto& pair : thisObject->moduleRecord()->importEntries()) {
            const AbstractModuleRecord::ImportEntry& importEntry = pair.value;
            if (importEntry.type == AbstractModuleRecord::ImportEntryType::Single)
                propertyNamesArray.add(importEntry.localName);
        }
    }
}

bool JSModuleEnvironment::put(JSCell* cell, JSGlobalObject* globalObject, PropertyName propertyName, JSValue value, PutPropertySlot& slot)
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    JSModuleEnvironment* thisObject = jsCast<JSModuleEnvironment*>(cell);
    // All imported bindings are immutable.
    AbstractModuleRecord::Resolution resolution = thisObject->moduleRecord()->resolveImport(globalObject, Identifier::fromUid(vm, propertyName.uid()));
    RETURN_IF_EXCEPTION(scope, false);
    if (resolution.type == AbstractModuleRecord::Resolution::Type::Resolved) {
        throwTypeError(globalObject, scope, ReadonlyPropertyWriteError);
        return false;
    }
    RELEASE_AND_RETURN(scope, Base::put(thisObject, globalObject, propertyName, value, slot));
}

bool JSModuleEnvironment::deleteProperty(JSCell* cell, JSGlobalObject* globalObject, PropertyName propertyName, DeletePropertySlot& slot)
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    JSModuleEnvironment* thisObject = jsCast<JSModuleEnvironment*>(cell);
    // All imported bindings are immutable.
    AbstractModuleRecord::Resolution resolution = thisObject->moduleRecord()->resolveImport(globalObject, Identifier::fromUid(vm, propertyName.uid()));
    RETURN_IF_EXCEPTION(scope, false);
    if (resolution.type == AbstractModuleRecord::Resolution::Type::Resolved)
        return false;
    return Base::deleteProperty(thisObject, globalObject, propertyName, slot);
}

} // namespace JSC
