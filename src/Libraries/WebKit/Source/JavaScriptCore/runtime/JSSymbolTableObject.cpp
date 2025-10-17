/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 2, 2022.
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
#include "JSSymbolTableObject.h"

#include "JSCInlines.h"
#include "PropertyNameArray.h"

namespace JSC {

const ClassInfo JSSymbolTableObject::s_info = { "SymbolTableObject"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(JSSymbolTableObject) };

template<typename Visitor>
void JSSymbolTableObject::visitChildrenImpl(JSCell* cell, Visitor& visitor)
{
    JSSymbolTableObject* thisObject = jsCast<JSSymbolTableObject*>(cell);
    ASSERT_GC_OBJECT_INHERITS(thisObject, info());
    Base::visitChildren(thisObject, visitor);
    visitor.append(thisObject->m_symbolTable);
}

DEFINE_VISIT_CHILDREN(JSSymbolTableObject);

bool JSSymbolTableObject::deleteProperty(JSCell* cell, JSGlobalObject* globalObject, PropertyName propertyName, DeletePropertySlot& slot)
{
    JSSymbolTableObject* thisObject = jsCast<JSSymbolTableObject*>(cell);
    if (thisObject->symbolTable()->contains(propertyName.uid()))
        return false;

    return Base::deleteProperty(thisObject, globalObject, propertyName, slot);
}

void JSSymbolTableObject::getOwnSpecialPropertyNames(JSObject* object, JSGlobalObject* globalObject, PropertyNameArray& propertyNames, DontEnumPropertiesMode mode)
{
    VM& vm = globalObject->vm();
    JSSymbolTableObject* thisObject = jsCast<JSSymbolTableObject*>(object);
    SymbolTable* symbolTable = thisObject->symbolTable();
    {
        ConcurrentJSLocker locker(symbolTable->m_lock);
        SymbolTable::Map::iterator end = symbolTable->end(locker);
        for (SymbolTable::Map::iterator it = symbolTable->begin(locker); it != end; ++it) {
            if (mode == DontEnumPropertiesMode::Include || !it->value.isDontEnum()) {
                if (!propertyNames.includeSymbolProperties() && it->key->isSymbol())
                    continue;
                if (propertyNames.privateSymbolMode() == PrivateSymbolMode::Exclude && symbolTable->hasPrivateName(it->key))
                    continue;
                propertyNames.add(Identifier::fromUid(vm, it->key.get()));
            }
        }
    }
}

} // namespace JSC
