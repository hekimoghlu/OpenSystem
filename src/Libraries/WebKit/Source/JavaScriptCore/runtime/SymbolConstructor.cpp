/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 27, 2022.
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
#include "SymbolConstructor.h"

#include "JSCInlines.h"
#include "SymbolPrototype.h"
#include <wtf/text/SymbolRegistry.h>

namespace JSC {

static JSC_DECLARE_HOST_FUNCTION(symbolConstructorFor);
static JSC_DECLARE_HOST_FUNCTION(symbolConstructorKeyFor);

}

#include "SymbolConstructor.lut.h"

namespace JSC {

STATIC_ASSERT_IS_TRIVIALLY_DESTRUCTIBLE(SymbolConstructor);

const ClassInfo SymbolConstructor::s_info = { "Function"_s, &Base::s_info, &symbolConstructorTable, nullptr, CREATE_METHOD_TABLE(SymbolConstructor) };

/* Source for SymbolConstructor.lut.h
@begin symbolConstructorTable
  for       symbolConstructorFor       DontEnum|Function 1
  keyFor    symbolConstructorKeyFor    DontEnum|Function 1
@end
*/

static JSC_DECLARE_HOST_FUNCTION(callSymbol);
static JSC_DECLARE_HOST_FUNCTION(constructSymbol);

SymbolConstructor::SymbolConstructor(VM& vm, Structure* structure)
    : InternalFunction(vm, structure, callSymbol, constructSymbol)
{
}

#define INITIALIZE_WELL_KNOWN_SYMBOLS(name) \
putDirectWithoutTransition(vm, Identifier::fromString(vm, #name ""_s), Symbol::create(vm, static_cast<SymbolImpl&>(*vm.propertyNames->name##Symbol.impl())), PropertyAttribute::DontEnum | PropertyAttribute::DontDelete | PropertyAttribute::ReadOnly);

void SymbolConstructor::finishCreation(VM& vm, SymbolPrototype* prototype)
{
    Base::finishCreation(vm, 0, vm.propertyNames->Symbol.string(), PropertyAdditionMode::WithoutStructureTransition);
    putDirectWithoutTransition(vm, vm.propertyNames->prototype, prototype, PropertyAttribute::DontEnum | PropertyAttribute::DontDelete | PropertyAttribute::ReadOnly);

    JSC_COMMON_PRIVATE_IDENTIFIERS_EACH_WELL_KNOWN_SYMBOL(INITIALIZE_WELL_KNOWN_SYMBOLS)
}

// ------------------------------ Functions ---------------------------

JSC_DEFINE_HOST_FUNCTION(callSymbol, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    JSValue description = callFrame->argument(0);
    if (description.isUndefined())
        return JSValue::encode(Symbol::create(vm));

    String string = description.toWTFString(globalObject);
    RETURN_IF_EXCEPTION(scope, { });
    return JSValue::encode(Symbol::createWithDescription(vm, string));
}

JSC_DEFINE_HOST_FUNCTION(constructSymbol, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);
    return throwVMError(globalObject, scope, createNotAConstructorError(globalObject, callFrame->jsCallee()));
}

JSC_DEFINE_HOST_FUNCTION(symbolConstructorFor, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    JSString* stringKey = callFrame->argument(0).toString(globalObject);
    RETURN_IF_EXCEPTION(scope, encodedJSValue());
    auto string = stringKey->value(globalObject);
    RETURN_IF_EXCEPTION(scope, encodedJSValue());

    return JSValue::encode(Symbol::create(vm, vm.symbolRegistry().symbolForKey(string)));
}

const ASCIILiteral SymbolKeyForTypeError { "Symbol.keyFor requires that the first argument be a symbol"_s };

JSC_DEFINE_HOST_FUNCTION(symbolConstructorKeyFor, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    JSValue symbolValue = callFrame->argument(0);
    if (!symbolValue.isSymbol())
        return JSValue::encode(throwTypeError(globalObject, scope, SymbolKeyForTypeError));

    PrivateName privateName = asSymbol(symbolValue)->privateName();
    SymbolImpl& uid = privateName.uid();
    if (!uid.symbolRegistry())
        return JSValue::encode(jsUndefined());

    ASSERT(uid.symbolRegistry() == &vm.symbolRegistry());
    return JSValue::encode(jsString(vm, String { uid }));
}

} // namespace JSC
