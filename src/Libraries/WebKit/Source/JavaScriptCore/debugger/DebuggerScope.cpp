/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 30, 2024.
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
#include "DebuggerScope.h"

#include "JSLexicalEnvironment.h"
#include "JSCInlines.h"

namespace JSC {

STATIC_ASSERT_IS_TRIVIALLY_DESTRUCTIBLE(DebuggerScope);

const ClassInfo DebuggerScope::s_info = { "DebuggerScope"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(DebuggerScope) };

DebuggerScope* DebuggerScope::create(VM& vm, JSScope* scope)
{
    Structure* structure = scope->globalObject()->debuggerScopeStructure();
    DebuggerScope* debuggerScope = new (NotNull, allocateCell<DebuggerScope>(vm)) DebuggerScope(vm, structure, scope);
    debuggerScope->finishCreation(vm);
    return debuggerScope;
}

DebuggerScope::DebuggerScope(VM& vm, Structure* structure, JSScope* scope)
    : JSNonFinalObject(vm, structure)
    , m_scope(scope, WriteBarrierEarlyInit)
{
    ASSERT(scope);
}

template<typename Visitor>
void DebuggerScope::visitChildrenImpl(JSCell* cell, Visitor& visitor)
{
    DebuggerScope* thisObject = jsCast<DebuggerScope*>(cell);
    ASSERT_GC_OBJECT_INHERITS(thisObject, info());
    Base::visitChildren(cell, visitor);

    visitor.append(thisObject->m_scope);
    visitor.append(thisObject->m_next);
}

DEFINE_VISIT_CHILDREN(DebuggerScope);

bool DebuggerScope::getOwnPropertySlot(JSObject* object, JSGlobalObject* globalObject, PropertyName propertyName, PropertySlot& slot)
{
    DebuggerScope* scope = jsCast<DebuggerScope*>(object);
    if (!scope->isValid())
        return false;
    JSObject* thisObject = JSScope::objectAtScope(scope->jsScope());
    slot.setThisValue(JSValue(thisObject));

    // By default, JSObject::getPropertySlot() will look in the DebuggerScope's prototype
    // chain and not the wrapped scope, and JSObject::getPropertySlot() cannot be overridden
    // to behave differently for the DebuggerScope.
    //
    // Instead, we'll treat all properties in the wrapped scope and its prototype chain as
    // the own properties of the DebuggerScope. This is fine because the WebInspector
    // does not presently need to distinguish between what's owned at each level in the
    // prototype chain. Hence, we'll invoke getPropertySlot() on the wrapped scope here
    // instead of getOwnPropertySlot().
    bool result = thisObject->getPropertySlot(globalObject, propertyName, slot);
    if (result && slot.isValue() && slot.getValue(globalObject, propertyName) == jsTDZValue()) {
        // FIXME:
        // We hit a scope property that has the TDZ empty value.
        // Currently, we just lie to the inspector and claim that this property is undefined.
        // This is not ideal and we should fix it.
        // https://bugs.webkit.org/show_bug.cgi?id=144977
        slot.setValue(slot.slotBase(), static_cast<unsigned>(PropertyAttribute::DontEnum), jsUndefined());
        return true;
    }
    return result;
}

bool DebuggerScope::put(JSCell* cell, JSGlobalObject* globalObject, PropertyName propertyName, JSValue value, PutPropertySlot& slot)
{
    DebuggerScope* scope = jsCast<DebuggerScope*>(cell);
    ASSERT(scope->isValid());
    if (!scope->isValid())
        return false;
    JSObject* thisObject = JSScope::objectAtScope(scope->jsScope());
    slot.setThisValue(JSValue(thisObject));
    return thisObject->methodTable()->put(thisObject, globalObject, propertyName, value, slot);
}

bool DebuggerScope::deleteProperty(JSCell* cell, JSGlobalObject* globalObject, PropertyName propertyName, DeletePropertySlot& slot)
{
    DebuggerScope* scope = jsCast<DebuggerScope*>(cell);
    ASSERT(scope->isValid());
    if (!scope->isValid())
        return false;
    JSObject* thisObject = JSScope::objectAtScope(scope->jsScope());
    return thisObject->methodTable()->deleteProperty(thisObject, globalObject, propertyName, slot);
}

void DebuggerScope::getOwnPropertyNames(JSObject* object, JSGlobalObject* globalObject, PropertyNameArray& propertyNames, DontEnumPropertiesMode mode)
{
    DebuggerScope* scope = jsCast<DebuggerScope*>(object);
    ASSERT(scope->isValid());
    if (!scope->isValid())
        return;
    JSObject* thisObject = JSScope::objectAtScope(scope->jsScope());
    thisObject->getPropertyNames(globalObject, propertyNames, mode);
}

bool DebuggerScope::defineOwnProperty(JSObject* object, JSGlobalObject* globalObject, PropertyName propertyName, const PropertyDescriptor& descriptor, bool shouldThrow)
{
    DebuggerScope* scope = jsCast<DebuggerScope*>(object);
    ASSERT(scope->isValid());
    if (!scope->isValid())
        return false;
    JSObject* thisObject = JSScope::objectAtScope(scope->jsScope());
    return thisObject->methodTable()->defineOwnProperty(thisObject, globalObject, propertyName, descriptor, shouldThrow);
}

DebuggerScope* DebuggerScope::next()
{
    ASSERT(isValid());
    if (!m_next && m_scope->next()) {
        VM& vm = m_scope->vm();
        DebuggerScope* nextScope = create(vm, m_scope->next());
        m_next.set(vm, this, nextScope);
    }
    return m_next.get();
}

void DebuggerScope::invalidateChain()
{
    if (!isValid())
        return;

    DebuggerScope* scope = this;
    while (scope) {
        DebuggerScope* nextScope = scope->m_next.get();
        scope->m_next.clear();
        scope->m_scope.clear(); // This also marks this scope as invalid.
        scope = nextScope;
    }
}

bool DebuggerScope::isCatchScope() const
{
    return m_scope->isCatchScope();
}

bool DebuggerScope::isFunctionNameScope() const
{
    return m_scope->isFunctionNameScopeObject();
}

bool DebuggerScope::isWithScope() const
{
    return m_scope->isWithScope();
}

bool DebuggerScope::isGlobalScope() const
{
    return m_scope->isGlobalObject();
}

bool DebuggerScope::isGlobalLexicalEnvironment() const
{
    return m_scope->isGlobalLexicalEnvironment();
}

bool DebuggerScope::isClosureScope() const
{
    // In the current debugger implementation, every function or eval will create an
    // lexical environment object. Hence, a lexical environment object implies a
    // function or eval scope.
    return m_scope->isVarScope() || m_scope->isLexicalScope();
}

bool DebuggerScope::isNestedLexicalScope() const
{
    return m_scope->isNestedLexicalScope();
}

String DebuggerScope::name() const
{
    SymbolTable* symbolTable = m_scope->symbolTable();
    if (!symbolTable)
        return String();

    CodeBlock* codeBlock = symbolTable->rareDataCodeBlock();
    if (!codeBlock)
        return String();

    return String::fromUTF8(codeBlock->inferredName().span());
}

DebuggerLocation DebuggerScope::location() const
{
    SymbolTable* symbolTable = m_scope->symbolTable();
    if (!symbolTable)
        return DebuggerLocation();

    CodeBlock* codeBlock = symbolTable->rareDataCodeBlock();
    if (!codeBlock)
        return DebuggerLocation();

    ScriptExecutable* executable = codeBlock->ownerExecutable();
    return DebuggerLocation(executable);
}

JSValue DebuggerScope::caughtValue(JSGlobalObject* globalObject) const
{
    ASSERT(isCatchScope());
    JSLexicalEnvironment* catchEnvironment = jsCast<JSLexicalEnvironment*>(m_scope.get());
    SymbolTable* catchSymbolTable = catchEnvironment->symbolTable();
    RELEASE_ASSERT(catchSymbolTable->size() == 1);
    PropertyName errorName(catchSymbolTable->begin(catchSymbolTable->m_lock)->key.get());
    PropertySlot slot(m_scope.get(), PropertySlot::InternalMethodType::Get);
    bool success = catchEnvironment->getOwnPropertySlot(catchEnvironment, globalObject, errorName, slot);
    RELEASE_ASSERT(success && slot.isValue());
    return slot.getValue(globalObject, errorName);
}

} // namespace JSC
