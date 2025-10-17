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
#include "WebAssemblyExceptionPrototype.h"

#if ENABLE(WEBASSEMBLY)

#include "AuxiliaryBarrierInlines.h"
#include "JSCInlines.h"
#include "JSWebAssemblyException.h"
#include "JSWebAssemblyHelpers.h"
#include "JSWebAssemblyTag.h"
#include <wtf/text/MakeString.h>

namespace JSC {
static JSC_DECLARE_HOST_FUNCTION(webAssemblyExceptionProtoFuncGetArg);
static JSC_DECLARE_HOST_FUNCTION(webAssemblyExceptionProtoFuncIs);
}

#include "WebAssemblyExceptionPrototype.lut.h"

namespace JSC {

const ClassInfo WebAssemblyExceptionPrototype::s_info = { "WebAssembly.Exception"_s, &Base::s_info, &prototypeTableWebAssemblyException, nullptr, CREATE_METHOD_TABLE(WebAssemblyExceptionPrototype) };

/* Source for WebAssemblyExceptionPrototype.lut.h
 @begin prototypeTableWebAssemblyException
 getArg   webAssemblyExceptionProtoFuncGetArg   Function 2
 is       webAssemblyExceptionProtoFuncIs       Function 1
 @end
 */

WebAssemblyExceptionPrototype* WebAssemblyExceptionPrototype::create(VM& vm, JSGlobalObject*, Structure* structure)
{
    auto* object = new (NotNull, allocateCell<WebAssemblyExceptionPrototype>(vm)) WebAssemblyExceptionPrototype(vm, structure);
    object->finishCreation(vm);
    return object;
}

Structure* WebAssemblyExceptionPrototype::createStructure(VM& vm, JSGlobalObject* globalObject, JSValue prototype)
{
    return Structure::create(vm, globalObject, prototype, TypeInfo(ObjectType, StructureFlags), info());
}

void WebAssemblyExceptionPrototype::finishCreation(VM& vm)
{
    Base::finishCreation(vm);
    ASSERT(inherits(info()));
    JSC_TO_STRING_TAG_WITHOUT_TRANSITION();
}

WebAssemblyExceptionPrototype::WebAssemblyExceptionPrototype(VM& vm, Structure* structure)
    : Base(vm, structure)
{
}

ALWAYS_INLINE static JSWebAssemblyException* getException(JSGlobalObject* globalObject, JSValue thisValue)
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    if (UNLIKELY(!thisValue.isCell())) {
        throwVMError(globalObject, scope, createNotAnObjectError(globalObject, thisValue));
        return nullptr;
    }
    auto* tag = jsDynamicCast<JSWebAssemblyException*>(thisValue.asCell());
    if (LIKELY(tag))
        return tag;
    throwTypeError(globalObject, scope, "WebAssembly.Exception operation called on non-Exception object"_s);
    return nullptr;
}

JSC_DEFINE_HOST_FUNCTION(webAssemblyExceptionProtoFuncGetArg, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto throwScope = DECLARE_THROW_SCOPE(vm);

    const auto& formatMessage = [&](const auto& message) {
        return makeString("WebAssembly.Exception.getArg(): "_s, message);
    };

    JSWebAssemblyException* jsException = getException(globalObject, callFrame->thisValue());
    RETURN_IF_EXCEPTION(throwScope, { });

    if (UNLIKELY(callFrame->argumentCount() < 2))
        return JSValue::encode(throwException(globalObject, throwScope, createNotEnoughArgumentsError(globalObject)));

    JSWebAssemblyTag* tag = jsDynamicCast<JSWebAssemblyTag*>(callFrame->argument(0));
    if (UNLIKELY(!tag))
        return throwVMTypeError(globalObject, throwScope, formatMessage("First argument must be a WebAssembly.Tag"_s));

    uint32_t index = toNonWrappingUint32(globalObject, callFrame->argument(1), ErrorType::RangeError);
    RETURN_IF_EXCEPTION(throwScope, { });

    if (UNLIKELY(jsException->tag() != tag->tag()))
        return throwVMTypeError(globalObject, throwScope, formatMessage("First argument does not match the exception tag"_s));

    if (UNLIKELY(index >= tag->tag().parameterCount()))
        return throwVMRangeError(globalObject, throwScope, formatMessage("Index out of range"_s));

    RELEASE_AND_RETURN(throwScope, JSValue::encode(jsException->getArg(globalObject, index)));
}

JSC_DEFINE_HOST_FUNCTION(webAssemblyExceptionProtoFuncIs, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto throwScope = DECLARE_THROW_SCOPE(vm);

    JSWebAssemblyException* jsException = getException(globalObject, callFrame->thisValue());
    RETURN_IF_EXCEPTION(throwScope, { });

    if (UNLIKELY(callFrame->argumentCount() < 1))
        return JSValue::encode(throwException(globalObject, throwScope, createNotEnoughArgumentsError(globalObject)));

    JSWebAssemblyTag* tag = jsDynamicCast<JSWebAssemblyTag*>(callFrame->argument(0));
    if (!tag)
        return throwVMTypeError(globalObject, throwScope, "WebAssembly.Exception.is(): First argument must be a WebAssembly.Tag"_s);

    RELEASE_AND_RETURN(throwScope, JSValue::encode(jsBoolean(jsException->tag() == tag->tag())));
}

} // namespace JSC

#endif // ENABLE(WEBASSEMBLY)
