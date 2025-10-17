/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 2, 2022.
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
#include "SymbolPrototype.h"

#include "IntegrityInlines.h"
#include "JSCInlines.h"
#include "SymbolObject.h"

namespace JSC {

static JSC_DECLARE_CUSTOM_GETTER(symbolProtoGetterDescription);
static JSC_DECLARE_HOST_FUNCTION(symbolProtoFuncToString);
static JSC_DECLARE_HOST_FUNCTION(symbolProtoFuncValueOf);

}

#include "SymbolPrototype.lut.h"

namespace JSC {

const ClassInfo SymbolPrototype::s_info = { "Symbol"_s, &Base::s_info, &symbolPrototypeTable, nullptr, CREATE_METHOD_TABLE(SymbolPrototype) };

/* Source for SymbolPrototype.lut.h
@begin symbolPrototypeTable
  description       symbolProtoGetterDescription    DontEnum|ReadOnly|CustomAccessor
  toString          symbolProtoFuncToString         DontEnum|Function 0
  valueOf           symbolProtoFuncValueOf          DontEnum|Function 0
@end
*/

SymbolPrototype::SymbolPrototype(VM& vm, Structure* structure)
    : Base(vm, structure)
{
}

void SymbolPrototype::finishCreation(VM& vm, JSGlobalObject* globalObject)
{
    Base::finishCreation(vm);
    ASSERT(inherits(info()));

    JSFunction* toPrimitiveFunction = JSFunction::create(vm, globalObject, 1, "[Symbol.toPrimitive]"_s, symbolProtoFuncValueOf, ImplementationVisibility::Public);
    putDirectWithoutTransition(vm, vm.propertyNames->toPrimitiveSymbol, toPrimitiveFunction, PropertyAttribute::DontEnum | PropertyAttribute::ReadOnly);
    JSC_TO_STRING_TAG_WITHOUT_TRANSITION();
}

// ------------------------------ Functions ---------------------------

static const ASCIILiteral SymbolDescriptionTypeError { "Symbol.prototype.description requires that |this| be a symbol or a symbol object"_s };
static const ASCIILiteral SymbolToStringTypeError { "Symbol.prototype.toString requires that |this| be a symbol or a symbol object"_s };
static const ASCIILiteral SymbolValueOfTypeError { "Symbol.prototype.valueOf requires that |this| be a symbol or a symbol object"_s };

inline Symbol* tryExtractSymbol(JSValue thisValue)
{
    if (thisValue.isSymbol())
        return asSymbol(thisValue);

    if (!thisValue.isObject())
        return nullptr;
    JSObject* thisObject = asObject(thisValue);
    if (!thisObject->inherits<SymbolObject>())
        return nullptr;
    return asSymbol(jsCast<SymbolObject*>(thisObject)->internalValue());
}

JSC_DEFINE_CUSTOM_GETTER(symbolProtoGetterDescription, (JSGlobalObject* globalObject, EncodedJSValue thisValue, PropertyName))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    Symbol* symbol = tryExtractSymbol(JSValue::decode(thisValue));
    if (!symbol)
        return throwVMTypeError(globalObject, scope, SymbolDescriptionTypeError);
    scope.release();
    Integrity::auditStructureID(symbol->structureID());
    auto description = symbol->description();
    return JSValue::encode(description.isNull() ? jsUndefined() : jsString(vm, WTFMove(description)));
}

JSC_DEFINE_HOST_FUNCTION(symbolProtoFuncToString, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    Symbol* symbol = tryExtractSymbol(callFrame->thisValue());
    if (!symbol)
        return throwVMTypeError(globalObject, scope, SymbolToStringTypeError);
    Integrity::auditStructureID(symbol->structureID());
    RELEASE_AND_RETURN(scope, JSValue::encode(jsNontrivialString(vm, symbol->descriptiveString())));
}

JSC_DEFINE_HOST_FUNCTION(symbolProtoFuncValueOf, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    Symbol* symbol = tryExtractSymbol(callFrame->thisValue());
    if (!symbol)
        return throwVMTypeError(globalObject, scope, SymbolValueOfTypeError);

    Integrity::auditStructureID(symbol->structureID());
    RELEASE_AND_RETURN(scope, JSValue::encode(symbol));
}

} // namespace JSC
