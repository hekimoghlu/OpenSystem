/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 1, 2025.
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
#include "WebAssemblyTablePrototype.h"

#if ENABLE(WEBASSEMBLY)

#include "AuxiliaryBarrierInlines.h"
#include "JSCJSValueInlines.h"
#include "JSGlobalObjectInlines.h"
#include "JSObjectInlines.h"
#include "JSWebAssemblyHelpers.h"
#include "JSWebAssemblyTable.h"
#include "StructureInlines.h"
#include "WasmTypeDefinition.h"

namespace JSC {
static JSC_DECLARE_CUSTOM_GETTER(webAssemblyTableProtoGetterLength);
static JSC_DECLARE_HOST_FUNCTION(webAssemblyTableProtoFuncGrow);
static JSC_DECLARE_HOST_FUNCTION(webAssemblyTableProtoFuncGet);
static JSC_DECLARE_HOST_FUNCTION(webAssemblyTableProtoFuncSet);
static JSC_DECLARE_HOST_FUNCTION(webAssemblyTableProtoFuncType);
}

#include "WebAssemblyTablePrototype.lut.h"

namespace JSC {

const ClassInfo WebAssemblyTablePrototype::s_info = { "WebAssembly.Table"_s, &Base::s_info, &prototypeTableWebAssemblyTable, nullptr, CREATE_METHOD_TABLE(WebAssemblyTablePrototype) };

/* Source for WebAssemblyTablePrototype.lut.h
 @begin prototypeTableWebAssemblyTable
 length webAssemblyTableProtoGetterLength ReadOnly|CustomAccessor
 grow   webAssemblyTableProtoFuncGrow   Function 1
 get    webAssemblyTableProtoFuncGet    Function 1
 set    webAssemblyTableProtoFuncSet    Function 1
 type   webAssemblyTableProtoFuncType   Function 0
 @end
 */

static ALWAYS_INLINE JSWebAssemblyTable* getTable(JSGlobalObject* globalObject, VM& vm, JSValue v)
{
    auto throwScope = DECLARE_THROW_SCOPE(vm);
    JSWebAssemblyTable* result = jsDynamicCast<JSWebAssemblyTable*>(v);
    if (!result) {
        throwException(globalObject, throwScope, 
            createTypeError(globalObject, "expected |this| value to be an instance of WebAssembly.Table"_s));
        return nullptr;
    }
    return result;
}

JSC_DEFINE_CUSTOM_GETTER(webAssemblyTableProtoGetterLength, (JSGlobalObject* globalObject, EncodedJSValue thisValue, PropertyName))
{
    VM& vm = globalObject->vm();
    auto throwScope = DECLARE_THROW_SCOPE(vm);

    JSWebAssemblyTable* table = getTable(globalObject, vm, JSValue::decode(thisValue));
    RETURN_IF_EXCEPTION(throwScope, encodedJSValue());
    return JSValue::encode(jsNumber(table->length()));
}

JSC_DEFINE_HOST_FUNCTION(webAssemblyTableProtoFuncGrow, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto throwScope = DECLARE_THROW_SCOPE(vm);

    JSWebAssemblyTable* table = getTable(globalObject, vm, callFrame->thisValue());
    RETURN_IF_EXCEPTION(throwScope, encodedJSValue());

    uint32_t delta = toNonWrappingUint32(globalObject, callFrame->argument(0));
    RETURN_IF_EXCEPTION(throwScope, encodedJSValue());

    JSValue defaultValue = jsNull();
    if (callFrame->argumentCount() < 2) {
        if (!table->table()->wasmType().isNullable())
            return throwVMTypeError(globalObject, throwScope, "WebAssembly.Table.prototype.grow requires the second argument for non-defaultable table type"_s);
        defaultValue = defaultValueForReferenceType(table->table()->wasmType());
    } else
        defaultValue = callFrame->uncheckedArgument(1);

    uint32_t oldLength = table->length();
    bool didGrow = !!table->grow(globalObject, delta, defaultValue);
    RETURN_IF_EXCEPTION(throwScope, encodedJSValue());

    if (!didGrow)
        return throwVMRangeError(globalObject, throwScope, "WebAssembly.Table.prototype.grow could not grow the table"_s);

    return JSValue::encode(jsNumber(oldLength));
}

JSC_DEFINE_HOST_FUNCTION(webAssemblyTableProtoFuncGet, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto throwScope = DECLARE_THROW_SCOPE(vm);

    JSWebAssemblyTable* table = getTable(globalObject, vm, callFrame->thisValue());
    RETURN_IF_EXCEPTION(throwScope, encodedJSValue());

    uint32_t index = toNonWrappingUint32(globalObject, callFrame->argument(0));
    RETURN_IF_EXCEPTION(throwScope, encodedJSValue());
    if (index >= table->length())
        return throwVMRangeError(globalObject, throwScope, "WebAssembly.Table.prototype.get expects an integer less than the length of the table"_s);

    RELEASE_AND_RETURN(throwScope, JSValue::encode(table->get(globalObject, index)));
}

JSC_DEFINE_HOST_FUNCTION(webAssemblyTableProtoFuncSet, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto throwScope = DECLARE_THROW_SCOPE(vm);

    JSWebAssemblyTable* table = getTable(globalObject, vm, callFrame->thisValue());
    RETURN_IF_EXCEPTION(throwScope, encodedJSValue());

    uint32_t index = toNonWrappingUint32(globalObject, callFrame->argument(0));
    RETURN_IF_EXCEPTION(throwScope, encodedJSValue());

    if (index >= table->length())
        return throwVMRangeError(globalObject, throwScope, "WebAssembly.Table.prototype.set expects an integer less than the length of the table"_s);

    JSValue value = callFrame->argument(1);
    if (callFrame->argumentCount() < 2) {
        value = defaultValueForReferenceType(table->table()->wasmType());
        if (!table->table()->wasmType().isNullable())
            return throwVMTypeError(globalObject, throwScope, "WebAssembly.Table.prototype.set requires the second argument for non-defaultable table type"_s);
    }

    throwScope.release();
    table->set(globalObject, index, value);
    return encodedJSUndefined();
}

JSC_DEFINE_HOST_FUNCTION(webAssemblyTableProtoFuncType, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto throwScope = DECLARE_THROW_SCOPE(vm);

    JSWebAssemblyTable* table = getTable(globalObject, vm, callFrame->thisValue());
    RETURN_IF_EXCEPTION(throwScope, encodedJSValue());
    JSObject* typeDescriptor = table->type(globalObject);
    if (!typeDescriptor)
        return throwVMTypeError(globalObject, throwScope, "WebAssembly.Table.prototype.type unable to produce type descriptor for the given table"_s);
    RELEASE_AND_RETURN(throwScope, JSValue::encode(typeDescriptor));
}

WebAssemblyTablePrototype* WebAssemblyTablePrototype::create(VM& vm, JSGlobalObject*, Structure* structure)
{
    auto* object = new (NotNull, allocateCell<WebAssemblyTablePrototype>(vm)) WebAssemblyTablePrototype(vm, structure);
    object->finishCreation(vm);
    return object;
}

Structure* WebAssemblyTablePrototype::createStructure(VM& vm, JSGlobalObject* globalObject, JSValue prototype)
{
    return Structure::create(vm, globalObject, prototype, TypeInfo(ObjectType, StructureFlags), info());
}

void WebAssemblyTablePrototype::finishCreation(VM& vm)
{
    Base::finishCreation(vm);
    ASSERT(inherits(info()));
    JSC_TO_STRING_TAG_WITHOUT_TRANSITION();
}

WebAssemblyTablePrototype::WebAssemblyTablePrototype(VM& vm, Structure* structure)
    : Base(vm, structure)
{
}

} // namespace JSC

#endif // ENABLE(WEBASSEMBLY)
