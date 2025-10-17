/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 21, 2025.
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
#include "WebAssemblyMemoryPrototype.h"

#if ENABLE(WEBASSEMBLY)

#include "AuxiliaryBarrierInlines.h"
#include "JSArrayBuffer.h"
#include "JSCJSValueInlines.h"
#include "JSGlobalObjectInlines.h"
#include "JSObjectInlines.h"
#include "JSWebAssemblyHelpers.h"
#include "JSWebAssemblyMemory.h"
#include "StructureInlines.h"

namespace JSC {
static JSC_DECLARE_HOST_FUNCTION(webAssemblyMemoryProtoFuncGrow);
static JSC_DECLARE_CUSTOM_GETTER(webAssemblyMemoryProtoGetterBuffer);
static JSC_DECLARE_HOST_FUNCTION(webAssemblyMemoryProtoFuncType);
}

#include "WebAssemblyMemoryPrototype.lut.h"


namespace JSC {

const ClassInfo WebAssemblyMemoryPrototype::s_info = { "WebAssembly.Memory"_s, &Base::s_info, &prototypeTableWebAssemblyMemory, nullptr, CREATE_METHOD_TABLE(WebAssemblyMemoryPrototype) };

/* Source for WebAssemblyMemoryPrototype.lut.h
@begin prototypeTableWebAssemblyMemory
 grow   webAssemblyMemoryProtoFuncGrow   Function 1
 buffer webAssemblyMemoryProtoGetterBuffer ReadOnly|CustomAccessor
 type   webAssemblyMemoryProtoFuncType   Function 0
@end
*/

ALWAYS_INLINE JSWebAssemblyMemory* getMemory(JSGlobalObject* globalObject, VM& vm, JSValue value)
{
    auto throwScope = DECLARE_THROW_SCOPE(vm);

    JSWebAssemblyMemory* memory = jsDynamicCast<JSWebAssemblyMemory*>(value); 
    if (!memory) {
        throwException(globalObject, throwScope, 
            createTypeError(globalObject, "WebAssembly.Memory.prototype.buffer getter called with non WebAssembly.Memory |this| value"_s));
        return nullptr;
    }
    return memory;
}

JSC_DEFINE_HOST_FUNCTION(webAssemblyMemoryProtoFuncGrow, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto throwScope = DECLARE_THROW_SCOPE(vm);

    JSWebAssemblyMemory* memory = getMemory(globalObject, vm, callFrame->thisValue()); 
    RETURN_IF_EXCEPTION(throwScope, { });
    
    uint32_t delta = toNonWrappingUint32(globalObject, callFrame->argument(0));
    RETURN_IF_EXCEPTION(throwScope, { });

    PageCount result = memory->grow(vm, globalObject, delta);
    RETURN_IF_EXCEPTION(throwScope, { });

    return JSValue::encode(jsNumber(result.pageCount()));
}

JSC_DEFINE_CUSTOM_GETTER(webAssemblyMemoryProtoGetterBuffer, (JSGlobalObject* globalObject, EncodedJSValue thisValue, PropertyName))
{
    VM& vm = globalObject->vm();
    auto throwScope = DECLARE_THROW_SCOPE(vm);

    JSWebAssemblyMemory* memory = getMemory(globalObject, vm, JSValue::decode(thisValue)); 
    RETURN_IF_EXCEPTION(throwScope, { });
    RELEASE_AND_RETURN(throwScope, JSValue::encode(memory->buffer(globalObject)));
}

JSC_DEFINE_HOST_FUNCTION(webAssemblyMemoryProtoFuncType, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto throwScope = DECLARE_THROW_SCOPE(vm);

    JSWebAssemblyMemory* memory = getMemory(globalObject, vm, callFrame->thisValue());
    RETURN_IF_EXCEPTION(throwScope, { });
    RELEASE_AND_RETURN(throwScope, JSValue::encode(memory->type(globalObject)));
}


WebAssemblyMemoryPrototype* WebAssemblyMemoryPrototype::create(VM& vm, JSGlobalObject*, Structure* structure)
{
    auto* object = new (NotNull, allocateCell<WebAssemblyMemoryPrototype>(vm)) WebAssemblyMemoryPrototype(vm, structure);
    object->finishCreation(vm);
    return object;
}

Structure* WebAssemblyMemoryPrototype::createStructure(VM& vm, JSGlobalObject* globalObject, JSValue prototype)
{
    return Structure::create(vm, globalObject, prototype, TypeInfo(ObjectType, StructureFlags), info());
}

void WebAssemblyMemoryPrototype::finishCreation(VM& vm)
{
    Base::finishCreation(vm);
    ASSERT(inherits(info()));
    JSC_TO_STRING_TAG_WITHOUT_TRANSITION();
}

WebAssemblyMemoryPrototype::WebAssemblyMemoryPrototype(VM& vm, Structure* structure)
    : Base(vm, structure)
{
}

} // namespace JSC

#endif // ENABLE(WEBASSEMBLY)
