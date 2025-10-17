/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 12, 2024.
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
#include "WebAssemblyTableConstructor.h"

#if ENABLE(WEBASSEMBLY)

#include "JSCJSValueInlines.h"
#include "JSGlobalObjectInlines.h"
#include "JSObjectInlines.h"
#include "JSWebAssemblyHelpers.h"
#include "JSWebAssemblyTable.h"
#include "StructureInlines.h"
#include "WebAssemblyTablePrototype.h"

namespace JSC {

const ClassInfo WebAssemblyTableConstructor::s_info = { "Function"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(WebAssemblyTableConstructor) };

static JSC_DECLARE_HOST_FUNCTION(callJSWebAssemblyTable);
static JSC_DECLARE_HOST_FUNCTION(constructJSWebAssemblyTable);

JSC_DEFINE_HOST_FUNCTION(constructJSWebAssemblyTable, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto throwScope = DECLARE_THROW_SCOPE(vm);

    JSObject* newTarget = asObject(callFrame->newTarget());
    Structure* webAssemblyTableStructure = JSC_GET_DERIVED_STRUCTURE(vm, webAssemblyTableStructure, newTarget, callFrame->jsCallee());
    RETURN_IF_EXCEPTION(throwScope, { });

    JSObject* memoryDescriptor;
    {
        JSValue argument = callFrame->argument(0);
        if (!argument.isObject())
            return throwVMTypeError(globalObject, throwScope, "WebAssembly.Table expects its first argument to be an object"_s);
        memoryDescriptor = jsCast<JSObject*>(argument);
    }

    Wasm::TableElementType type;
    {
        Identifier elementIdent = Identifier::fromString(vm, "element"_s);
        JSValue elementValue = memoryDescriptor->get(globalObject, elementIdent);
        RETURN_IF_EXCEPTION(throwScope, encodedJSValue());
        String elementString = elementValue.toWTFString(globalObject);
        RETURN_IF_EXCEPTION(throwScope, encodedJSValue());
        if (elementString == "funcref"_s || elementString == "anyfunc"_s)
            type = Wasm::TableElementType::Funcref;
        else if (elementString == "externref"_s)
            type = Wasm::TableElementType::Externref;
        else
            return throwVMTypeError(globalObject, throwScope, "WebAssembly.Table expects its 'element' field to be the string 'funcref' or 'externref'"_s);
    }

    Identifier initialIdent = Identifier::fromString(vm, "initial"_s);
    JSValue initialSizeValue = memoryDescriptor->get(globalObject, initialIdent);
    RETURN_IF_EXCEPTION(throwScope, encodedJSValue());
    Identifier minimumIdent = Identifier::fromString(vm, "minimum"_s);
    JSValue minSizeValue = memoryDescriptor->get(globalObject, minimumIdent);
    RETURN_IF_EXCEPTION(throwScope, encodedJSValue());
    if (!initialSizeValue.isUndefined() && !minSizeValue.isUndefined())
        return throwVMTypeError(globalObject, throwScope, "WebAssembly.Table 'initial' and 'minimum' options are specified at the same time"_s);

    if (!minSizeValue.isUndefined())
        initialSizeValue = minSizeValue;

    uint32_t initial = toNonWrappingUint32(globalObject, initialSizeValue);
    RETURN_IF_EXCEPTION(throwScope, encodedJSValue());

    // In WebIDL, "present" means that [[Get]] result is undefined, not [[HasProperty]] result.
    // https://webidl.spec.whatwg.org/#idl-dictionaries
    std::optional<uint32_t> maximum;
    Identifier maximumIdent = Identifier::fromString(vm, "maximum"_s);
    JSValue maxSizeValue = memoryDescriptor->get(globalObject, maximumIdent);
    RETURN_IF_EXCEPTION(throwScope, encodedJSValue());
    if (!maxSizeValue.isUndefined()) {
        maximum = toNonWrappingUint32(globalObject, maxSizeValue);
        RETURN_IF_EXCEPTION(throwScope, encodedJSValue());

        if (initial > *maximum)
            return throwVMRangeError(globalObject, throwScope, "'maximum' property must be greater than or equal to the 'initial' property"_s);
    }

    RefPtr<Wasm::Table> wasmTable = Wasm::Table::tryCreate(initial, maximum, type, type == Wasm::TableElementType::Funcref ? Wasm::funcrefType() : Wasm::externrefType());
    if (!wasmTable)
        return throwVMRangeError(globalObject, throwScope, "couldn't create Table"_s);

    JSWebAssemblyTable* jsWebAssemblyTable = JSWebAssemblyTable::create(vm, webAssemblyTableStructure, wasmTable.releaseNonNull());

    JSValue defaultValue = callFrame->argumentCount() < 2
        ? defaultValueForReferenceType(jsWebAssemblyTable->table()->wasmType())
        : callFrame->uncheckedArgument(1);
    WebAssemblyFunction* wasmFunction = nullptr;
    WebAssemblyWrapperFunction* wasmWrapperFunction = nullptr;
    if (jsWebAssemblyTable->table()->isFuncrefTable() && !defaultValue.isNull() && !isWebAssemblyHostFunction(defaultValue, wasmFunction, wasmWrapperFunction))
        return throwVMTypeError(globalObject, throwScope, "WebAssembly.Table.prototype.constructor expects the second argument to be null or an instance of WebAssembly.Function"_s);
    for (uint32_t tableIndex = 0; tableIndex < initial; ++tableIndex) {
        if (jsWebAssemblyTable->table()->isFuncrefTable() && wasmFunction)
            jsWebAssemblyTable->set(tableIndex, wasmFunction);
        if (jsWebAssemblyTable->table()->isExternrefTable())
            jsWebAssemblyTable->set(tableIndex, defaultValue);
        RETURN_IF_EXCEPTION(throwScope, encodedJSValue());
    }

    RELEASE_AND_RETURN(throwScope, JSValue::encode(jsWebAssemblyTable));
}

JSC_DEFINE_HOST_FUNCTION(callJSWebAssemblyTable, (JSGlobalObject* globalObject, CallFrame*))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);
    return JSValue::encode(throwConstructorCannotBeCalledAsFunctionTypeError(globalObject, scope, "WebAssembly.Table"_s));
}

WebAssemblyTableConstructor* WebAssemblyTableConstructor::create(VM& vm, Structure* structure, WebAssemblyTablePrototype* thisPrototype)
{
    auto* constructor = new (NotNull, allocateCell<WebAssemblyTableConstructor>(vm)) WebAssemblyTableConstructor(vm, structure);
    constructor->finishCreation(vm, thisPrototype);
    return constructor;
}

Structure* WebAssemblyTableConstructor::createStructure(VM& vm, JSGlobalObject* globalObject, JSValue prototype)
{
    return Structure::create(vm, globalObject, prototype, TypeInfo(InternalFunctionType, StructureFlags), info());
}

void WebAssemblyTableConstructor::finishCreation(VM& vm, WebAssemblyTablePrototype* prototype)
{
    Base::finishCreation(vm, 1, "Table"_s, PropertyAdditionMode::WithoutStructureTransition);
    putDirectWithoutTransition(vm, vm.propertyNames->prototype, prototype, PropertyAttribute::DontEnum | PropertyAttribute::DontDelete | PropertyAttribute::ReadOnly);
}

WebAssemblyTableConstructor::WebAssemblyTableConstructor(VM& vm, Structure* structure)
    : Base(vm, structure, callJSWebAssemblyTable, constructJSWebAssemblyTable)
{
}

} // namespace JSC

#endif // ENABLE(WEBASSEMBLY)

