/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 1, 2023.
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
#include "JSLexicalEnvironment.h"

#include "HeapAnalyzer.h"
#include "JSCInlines.h"

namespace JSC {

const ClassInfo JSLexicalEnvironment::s_info = { "JSLexicalEnvironment"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(JSLexicalEnvironment) };

template<typename Visitor>
void JSLexicalEnvironment::visitChildrenImpl(JSCell* cell, Visitor& visitor)
{
    auto* thisObject = jsCast<JSLexicalEnvironment*>(cell);
    ASSERT_GC_OBJECT_INHERITS(thisObject, info());
    Base::visitChildren(thisObject, visitor);
    visitor.appendValuesHidden(thisObject->variables(), thisObject->symbolTable()->scopeSize());
}

DEFINE_VISIT_CHILDREN(JSLexicalEnvironment);

void JSLexicalEnvironment::analyzeHeap(JSCell* cell, HeapAnalyzer& analyzer)
{
    auto* thisObject = jsCast<JSLexicalEnvironment*>(cell);
    Base::analyzeHeap(cell, analyzer);

    ConcurrentJSLocker locker(thisObject->symbolTable()->m_lock);
    SymbolTable::Map::iterator end = thisObject->symbolTable()->end(locker);
    for (SymbolTable::Map::iterator it = thisObject->symbolTable()->begin(locker); it != end; ++it) {
        SymbolTableEntry::Fast entry = it->value;
        ASSERT(!entry.isNull());
        ScopeOffset offset = entry.scopeOffset();
        if (!thisObject->isValidScopeOffset(offset))
            continue;

        JSValue toValue = thisObject->variableAt(offset).get();
        if (toValue && toValue.isCell())
            analyzer.analyzeVariableNameEdge(thisObject, toValue.asCell(), it->key.get());
    }
}

void JSLexicalEnvironment::getOwnSpecialPropertyNames(JSObject* object, JSGlobalObject* globalObject, PropertyNameArray& propertyNames, DontEnumPropertiesMode mode)
{
    JSLexicalEnvironment* thisObject = jsCast<JSLexicalEnvironment*>(object);
    SymbolTable* symbolTable = thisObject->symbolTable();

    {
        ConcurrentJSLocker locker(symbolTable->m_lock);
        SymbolTable::Map::iterator end = symbolTable->end(locker);
        VM& vm = globalObject->vm();
        for (SymbolTable::Map::iterator it = symbolTable->begin(locker); it != end; ++it) {
            if (mode == DontEnumPropertiesMode::Exclude && it->value.isDontEnum())
                continue;
            if (!thisObject->isValidScopeOffset(it->value.scopeOffset()))
                continue;
            if (!propertyNames.includeSymbolProperties() && it->key->isSymbol())
                continue;
            if (propertyNames.privateSymbolMode() == PrivateSymbolMode::Exclude && symbolTable->hasPrivateName(it->key))
                continue;
            propertyNames.add(Identifier::fromUid(vm, it->key.get()));
        }
    }
}

bool JSLexicalEnvironment::getOwnPropertySlot(JSObject* object, JSGlobalObject* globalObject, PropertyName propertyName, PropertySlot& slot)
{
    JSLexicalEnvironment* thisObject = jsCast<JSLexicalEnvironment*>(object);

    if (symbolTableGet(thisObject, propertyName, slot))
        return true;

    VM& vm = globalObject->vm();
    unsigned attributes;
    if (JSValue value = thisObject->getDirect(vm, propertyName, attributes)) {
        RELEASE_ASSERT(!(attributes & PropertyAttribute::Accessor));
        slot.setValue(thisObject, attributes, value);
        return true;
    }

    // We don't call through to JSObject because there's no way to give a 
    // lexical environment object getter properties or a prototype.
    ASSERT(!thisObject->structure()->hasAnyKindOfGetterSetterProperties());
    ASSERT(thisObject->getPrototypeDirect().isNull());
    return false;
}

bool JSLexicalEnvironment::put(JSCell* cell, JSGlobalObject* globalObject, PropertyName propertyName, JSValue value, PutPropertySlot& slot)
{
    JSLexicalEnvironment* thisObject = jsCast<JSLexicalEnvironment*>(cell);
    ASSERT(!Heap::heap(value) || Heap::heap(value) == Heap::heap(thisObject));

    bool shouldThrowReadOnlyError = slot.isStrictMode() || thisObject->isLexicalScope();
    bool ignoreReadOnlyErrors = false;
    bool putResult = false;
    if (symbolTablePutInvalidateWatchpointSet(thisObject, globalObject, propertyName, value, shouldThrowReadOnlyError, ignoreReadOnlyErrors, putResult))
        return putResult;

    // We don't call through to JSObject because __proto__ and getter/setter 
    // properties are non-standard extensions that other implementations do not
    // expose in the lexicalEnvironment object.
    ASSERT(!thisObject->structure()->hasAnyKindOfGetterSetterProperties());
    return thisObject->putOwnDataProperty(globalObject->vm(), propertyName, value, slot);
}

bool JSLexicalEnvironment::deleteProperty(JSCell* cell, JSGlobalObject* globalObject, PropertyName propertyName, DeletePropertySlot& slot)
{
    VM& vm = globalObject->vm();
    if (propertyName == vm.propertyNames->arguments)
        return false;

    return Base::deleteProperty(cell, globalObject, propertyName, slot);
}

} // namespace JSC
