/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 23, 2025.
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
#include "WebAssemblyMemoryConstructor.h"

#if ENABLE(WEBASSEMBLY)

#include "JSCJSValueInlines.h"
#include "JSGlobalObjectInlines.h"
#include "JSObjectInlines.h"
#include "JSWebAssemblyHelpers.h"
#include "JSWebAssemblyMemory.h"
#include "PageCount.h"
#include "StructureInlines.h"
#include "WasmMemory.h"
#include "WebAssemblyMemoryPrototype.h"

namespace JSC {

const ClassInfo WebAssemblyMemoryConstructor::s_info = { "Function"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(WebAssemblyMemoryConstructor) };

static JSC_DECLARE_HOST_FUNCTION(constructJSWebAssemblyMemory);
static JSC_DECLARE_HOST_FUNCTION(callJSWebAssemblyMemory);

JSWebAssemblyMemory* WebAssemblyMemoryConstructor::createMemoryFromDescriptor(JSGlobalObject* globalObject, Structure* webAssemblyMemoryStructure, JSObject* memoryDescriptor, std::optional<MemoryMode> desiredMemoryMode)
{
    VM& vm = globalObject->vm();
    auto throwScope = DECLARE_THROW_SCOPE(vm);

    PageCount initialPageCount;
    {
        Identifier initial = Identifier::fromString(vm, "initial"_s);
        JSValue initSizeValue = memoryDescriptor->get(globalObject, initial);
        RETURN_IF_EXCEPTION(throwScope, { });
        Identifier minimum = Identifier::fromString(vm, "minimum"_s);
        JSValue minSizeValue = memoryDescriptor->get(globalObject, minimum);
        RETURN_IF_EXCEPTION(throwScope, { });
        if (!minSizeValue.isUndefined() && !initSizeValue.isUndefined()) {
            // Error because both specified.
            throwTypeError(globalObject, throwScope, "WebAssembly.Memory 'initial' and 'minimum' options are specified at the same time"_s);
            return { };
        }
        if (!initSizeValue.isUndefined())
            minSizeValue = initSizeValue;

        uint32_t size = toNonWrappingUint32(globalObject, minSizeValue);
        RETURN_IF_EXCEPTION(throwScope, { });
        if (!PageCount::isValid(size)) {
            throwException(globalObject, throwScope, createRangeError(globalObject, "WebAssembly.Memory 'initial' page count is too large"_s));
            return { };
        }
        if (PageCount(size).bytes() > MAX_ARRAY_BUFFER_SIZE) {
            throwException(globalObject, throwScope, createOutOfMemoryError(globalObject));
            return { };
        }
        initialPageCount = PageCount(size);
    }

    PageCount maximumPageCount;
    {
        // In WebIDL, "present" means that [[Get]] result is undefined, not [[HasProperty]] result.
        // https://webidl.spec.whatwg.org/#idl-dictionaries
        Identifier maximum = Identifier::fromString(vm, "maximum"_s);
        JSValue maxSizeValue = memoryDescriptor->get(globalObject, maximum);
        RETURN_IF_EXCEPTION(throwScope, { });
        if (!maxSizeValue.isUndefined()) {
            uint32_t size = toNonWrappingUint32(globalObject, maxSizeValue);
            RETURN_IF_EXCEPTION(throwScope, { });
            if (!PageCount::isValid(size)) {
                throwException(globalObject, throwScope, createRangeError(globalObject, "WebAssembly.Memory 'maximum' page count is too large"_s));
                return { };
            }
            maximumPageCount = PageCount(size);

            if (initialPageCount > maximumPageCount) {
                throwException(globalObject, throwScope,
                    createRangeError(globalObject, "'maximum' page count must be than greater than or equal to the 'initial' page count"_s));
                return { };
            }
        }
    }

    // Even though Options::useSharedArrayBuffer() is false, we can create SharedArrayBuffer through wasm shared memory.
    // But we cannot send SharedArrayBuffer to the other workers, so it is not effective.
    MemorySharingMode sharingMode = MemorySharingMode::Default;
    if (LIKELY(Options::useWasmFaultSignalHandler())) {
        JSValue sharedValue = memoryDescriptor->get(globalObject, Identifier::fromString(vm, "shared"_s));
        RETURN_IF_EXCEPTION(throwScope, { });
        bool shared = sharedValue.toBoolean(globalObject);
        RETURN_IF_EXCEPTION(throwScope, { });
        if (shared) {
            if (!maximumPageCount) {
                throwTypeError(globalObject, throwScope, "'maximum' page count must be defined if 'shared' is true"_s);
                return { };
            }
            sharingMode = MemorySharingMode::Shared;
        }
    }

    auto* jsMemory = JSWebAssemblyMemory::create(vm, webAssemblyMemoryStructure);

    RefPtr<Wasm::Memory> memory = Wasm::Memory::tryCreate(vm, initialPageCount, maximumPageCount, sharingMode, desiredMemoryMode,
        [&vm, jsMemory] (Wasm::Memory::GrowSuccess, PageCount oldPageCount, PageCount newPageCount) { jsMemory->growSuccessCallback(vm, oldPageCount, newPageCount); });
    if (!memory) {
        throwException(globalObject, throwScope, createOutOfMemoryError(globalObject));
        return { };
    }

    jsMemory->adopt(memory.releaseNonNull());
    return jsMemory;
}

JSC_DEFINE_HOST_FUNCTION(constructJSWebAssemblyMemory, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto throwScope = DECLARE_THROW_SCOPE(vm);

    JSObject* newTarget = asObject(callFrame->newTarget());
    Structure* webAssemblyMemoryStructure = JSC_GET_DERIVED_STRUCTURE(vm, webAssemblyMemoryStructure, newTarget, callFrame->jsCallee());
    RETURN_IF_EXCEPTION(throwScope, { });

    JSObject* memoryDescriptor;
    {
        JSValue argument = callFrame->argument(0);
        if (!argument.isObject())
            return JSValue::encode(throwException(globalObject, throwScope, createTypeError(globalObject, "WebAssembly.Memory expects its first argument to be an object"_s)));
        memoryDescriptor = jsCast<JSObject*>(argument);
    }

    JSWebAssemblyMemory* memory = WebAssemblyMemoryConstructor::createMemoryFromDescriptor(globalObject, webAssemblyMemoryStructure, memoryDescriptor);
    RETURN_IF_EXCEPTION(throwScope, encodedJSValue());

    return JSValue::encode(memory);
}

JSC_DEFINE_HOST_FUNCTION(callJSWebAssemblyMemory, (JSGlobalObject* globalObject, CallFrame*))
{
    VM& vm = globalObject->vm();
    auto throwScope = DECLARE_THROW_SCOPE(vm);
    return JSValue::encode(throwConstructorCannotBeCalledAsFunctionTypeError(globalObject, throwScope, "WebAssembly.Memory"_s));
}

WebAssemblyMemoryConstructor* WebAssemblyMemoryConstructor::create(VM& vm, Structure* structure, WebAssemblyMemoryPrototype* thisPrototype)
{
    auto* constructor = new (NotNull, allocateCell<WebAssemblyMemoryConstructor>(vm)) WebAssemblyMemoryConstructor(vm, structure);
    constructor->finishCreation(vm, thisPrototype);
    return constructor;
}

Structure* WebAssemblyMemoryConstructor::createStructure(VM& vm, JSGlobalObject* globalObject, JSValue prototype)
{
    return Structure::create(vm, globalObject, prototype, TypeInfo(InternalFunctionType, StructureFlags), info());
}

void WebAssemblyMemoryConstructor::finishCreation(VM& vm, WebAssemblyMemoryPrototype* prototype)
{
    Base::finishCreation(vm, 1, "Memory"_s, PropertyAdditionMode::WithoutStructureTransition);
    putDirectWithoutTransition(vm, vm.propertyNames->prototype, prototype, PropertyAttribute::DontEnum | PropertyAttribute::DontDelete | PropertyAttribute::ReadOnly);
}

WebAssemblyMemoryConstructor::WebAssemblyMemoryConstructor(VM& vm, Structure* structure)
    : Base(vm, structure, callJSWebAssemblyMemory, constructJSWebAssemblyMemory)
{
}

} // namespace JSC

#endif // ENABLE(WEBASSEMBLY)

