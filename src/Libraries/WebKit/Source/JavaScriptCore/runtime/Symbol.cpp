/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 19, 2022.
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
#include "Symbol.h"

#include "JSCJSValueInlines.h"
#include "SymbolObject.h"
#include <wtf/text/MakeString.h>

namespace JSC {

const ClassInfo Symbol::s_info = { "symbol"_s, nullptr, nullptr, nullptr, CREATE_METHOD_TABLE(Symbol) };

Symbol::Symbol(VM& vm)
    : Base(vm, vm.symbolStructure.get())
    , m_privateName(SymbolImpl::createNullSymbol())
{
}

Symbol::Symbol(VM& vm, const String& string)
    : Base(vm, vm.symbolStructure.get())
    , m_privateName(PrivateName::Description, string)
{
}

Symbol::Symbol(VM& vm, SymbolImpl& uid)
    : Base(vm, vm.symbolStructure.get())
    , m_privateName(uid)
{
}

void Symbol::finishCreation(VM& vm)
{
    Base::finishCreation(vm);
    ASSERT(inherits(info()));

    vm.symbolImplToSymbolMap.set(&m_privateName.uid(), this);
}

JSValue Symbol::toPrimitive(JSGlobalObject*, PreferredPrimitiveType) const
{
    return const_cast<Symbol*>(this);
}

JSObject* Symbol::toObject(JSGlobalObject* globalObject) const
{
    return SymbolObject::create(globalObject->vm(), globalObject->symbolObjectStructure(), const_cast<Symbol*>(this));
}

double Symbol::toNumber(JSGlobalObject* globalObject) const
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);
    throwTypeError(globalObject, scope, "Cannot convert a symbol to a number"_s);
    return 0.0;
}

void Symbol::destroy(JSCell* cell)
{
    static_cast<Symbol*>(cell)->Symbol::~Symbol();
}

String Symbol::descriptiveString() const
{
    return makeString("Symbol("_s, StringView(m_privateName.uid()), ')');
}

Expected<String, ErrorTypeWithExtension> Symbol::tryGetDescriptiveString() const
{
    String description = tryMakeString("Symbol("_s, StringView(m_privateName.uid()), ')');
    if (!description)
        return makeUnexpected(ErrorTypeWithExtension::OutOfMemoryError);
    return description;
}

String Symbol::description() const
{
    auto& uid = m_privateName.uid();
    return uid.isNullSymbol() ? String() : uid;
}

Symbol* Symbol::create(VM& vm)
{
    Symbol* symbol = new (NotNull, allocateCell<Symbol>(vm)) Symbol(vm);
    symbol->finishCreation(vm);
    return symbol;
}

Symbol* Symbol::createWithDescription(VM& vm, const String& description)
{
    Symbol* symbol = new (NotNull, allocateCell<Symbol>(vm)) Symbol(vm, description);
    symbol->finishCreation(vm);
    return symbol;
}

Symbol* Symbol::create(VM& vm, SymbolImpl& uid)
{
    if (Symbol* symbol = vm.symbolImplToSymbolMap.get(&uid))
        return symbol;

    Symbol* symbol = new (NotNull, allocateCell<Symbol>(vm)) Symbol(vm, uid);
    symbol->finishCreation(vm);
    return symbol;
}

} // namespace JSC
