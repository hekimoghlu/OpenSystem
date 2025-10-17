/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 28, 2022.
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
#include "JSSegmentedVariableObject.h"

#include "HeapAnalyzer.h"
#include "JSCInlines.h"

namespace JSC {

const ClassInfo JSSegmentedVariableObject::s_info = { "SegmentedVariableObject"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(JSSegmentedVariableObject) };

ScopeOffset JSSegmentedVariableObject::findVariableIndex(void* variableAddress)
{
    Locker locker { cellLock() };
    
    for (unsigned i = m_variables.size(); i--;) {
        if (&m_variables[i] != variableAddress)
            continue;
        return ScopeOffset(i);
    }
    CRASH();
    return ScopeOffset();
}

ScopeOffset JSSegmentedVariableObject::addVariables(unsigned numberOfVariablesToAdd, JSValue initialValue)
{
    Locker locker { cellLock() };
    
    size_t oldSize = m_variables.size();
    m_variables.grow(oldSize + numberOfVariablesToAdd);
    
    for (size_t i = numberOfVariablesToAdd; i--;)
        m_variables[oldSize + i].setWithoutWriteBarrier(initialValue);
    
    return ScopeOffset(oldSize);
}

template<typename Visitor>
void JSSegmentedVariableObject::visitChildrenImpl(JSCell* cell, Visitor& visitor)
{
    JSSegmentedVariableObject* thisObject = jsCast<JSSegmentedVariableObject*>(cell);
    ASSERT_GC_OBJECT_INHERITS(thisObject, info());
    Base::visitChildren(thisObject, visitor);
    
    // FIXME: We could avoid locking here if SegmentedVector was lock-free. It could be made lock-free
    // relatively easily.
    Locker locker { thisObject->cellLock() };
    for (unsigned i = thisObject->m_variables.size(); i--;)
        visitor.appendHidden(thisObject->m_variables[i]);
}

DEFINE_VISIT_CHILDREN_WITH_MODIFIER(JS_EXPORT_PRIVATE, JSSegmentedVariableObject);

void JSSegmentedVariableObject::analyzeHeap(JSCell* cell, HeapAnalyzer& analyzer)
{
    JSSegmentedVariableObject* thisObject = jsCast<JSSegmentedVariableObject*>(cell);
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

JSSegmentedVariableObject::JSSegmentedVariableObject(VM& vm, Structure* structure, JSScope* scope)
    : JSSymbolTableObject(vm, structure, scope)
{
}

JSSegmentedVariableObject::~JSSegmentedVariableObject()
{
#ifndef NDEBUG
    ASSERT(!m_alreadyDestroyed);
    m_alreadyDestroyed = true;
#endif
}

void JSSegmentedVariableObject::finishCreation(VM& vm)
{
    Base::finishCreation(vm);
    setSymbolTable(vm, SymbolTable::create(vm));
}

} // namespace JSC

