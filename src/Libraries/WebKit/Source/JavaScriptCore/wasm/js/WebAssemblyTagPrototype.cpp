/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 9, 2023.
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
#include "WebAssemblyTagPrototype.h"

#if ENABLE(WEBASSEMBLY)

#include "JSCellInlines.h"
#include "JSObjectInlines.h"
#include "JSWebAssemblyTag.h"
#include "ObjectConstructor.h"
#include "StructureInlines.h"
#include "WasmFormat.h"

namespace JSC {
static JSC_DECLARE_HOST_FUNCTION(webAssemblyTagProtoFuncType);
}

#include "WebAssemblyTagPrototype.lut.h"

namespace JSC {

const ClassInfo WebAssemblyTagPrototype::s_info = { "WebAssembly.Tag"_s, &Base::s_info, &prototypeTableWebAssemblyTag, nullptr, CREATE_METHOD_TABLE(WebAssemblyTagPrototype) };

/* Source for WebAssemblyTagPrototype.lut.h
 @begin prototypeTableWebAssemblyTag
 type   webAssemblyTagProtoFuncType   Function 0
 @end
 */


WebAssemblyTagPrototype* WebAssemblyTagPrototype::create(VM& vm, JSGlobalObject*, Structure* structure)
{
    auto* object = new (NotNull, allocateCell<WebAssemblyTagPrototype>(vm)) WebAssemblyTagPrototype(vm, structure);
    object->finishCreation(vm);
    return object;
}

Structure* WebAssemblyTagPrototype::createStructure(VM& vm, JSGlobalObject* globalObject, JSValue prototype)
{
    return Structure::create(vm, globalObject, prototype, TypeInfo(ObjectType, StructureFlags), info());
}

void WebAssemblyTagPrototype::finishCreation(VM& vm)
{
    Base::finishCreation(vm);
    ASSERT(inherits(info()));
    JSC_TO_STRING_TAG_WITHOUT_TRANSITION();
}

WebAssemblyTagPrototype::WebAssemblyTagPrototype(VM& vm, Structure* structure)
    : Base(vm, structure)
{
}

ALWAYS_INLINE static JSWebAssemblyTag* getTag(JSGlobalObject* globalObject, JSValue thisValue)
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    if (UNLIKELY(!thisValue.isCell())) {
        throwVMError(globalObject, scope, createNotAnObjectError(globalObject, thisValue));
        return nullptr;
    }
    auto* tag = jsDynamicCast<JSWebAssemblyTag*>(thisValue.asCell());
    if (LIKELY(tag))
        return tag;
    throwTypeError(globalObject, scope, "WebAssembly.Tag operation called on non-Tag object"_s);
    return nullptr;
}

JSC_DEFINE_HOST_FUNCTION(webAssemblyTagProtoFuncType, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto throwScope = DECLARE_THROW_SCOPE(vm);

    JSWebAssemblyTag* jsTag = getTag(globalObject, callFrame->thisValue());
    RETURN_IF_EXCEPTION(throwScope, encodedJSValue());

    const Wasm::Tag& tag = jsTag->tag();

    MarkedArgumentBuffer argList;
    argList.ensureCapacity(tag.parameterCount());
    for (size_t i = 0; i < tag.parameterCount(); ++i) {
        JSString* valueString = typeToJSAPIString(vm, tag.parameter(i));
        if (!valueString)
            return throwVMTypeError(globalObject, throwScope, "WebAssembly.Tag.prototype.type unable to produce type descriptor for the given tag"_s);
        argList.append(valueString);
    }

    if (UNLIKELY(argList.hasOverflowed())) {
        throwOutOfMemoryError(globalObject, throwScope);
        return encodedJSValue();
    }

    JSArray* parameters = constructArray(globalObject, static_cast<ArrayAllocationProfile*>(nullptr), argList);
    JSObject* type = constructEmptyObject(globalObject, globalObject->objectPrototype(), 1);
    type->putDirect(vm, Identifier::fromString(vm, "parameters"_s), parameters);

    RETURN_IF_EXCEPTION(throwScope, encodedJSValue());
    RELEASE_AND_RETURN(throwScope, JSValue::encode(type));
}

} // namespace JSC

#endif // ENABLE(WEBASSEMBLY)
