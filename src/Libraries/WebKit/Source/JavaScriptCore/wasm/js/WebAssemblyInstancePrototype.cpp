/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 10, 2022.
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
#include "WebAssemblyInstancePrototype.h"

#if ENABLE(WEBASSEMBLY)

#include "AuxiliaryBarrierInlines.h"
#include "JSCInlines.h"
#include "JSModuleNamespaceObject.h"
#include "JSWebAssemblyInstance.h"
#include "WebAssemblyModuleRecord.h"

namespace JSC {
static JSC_DECLARE_HOST_FUNCTION(webAssemblyInstanceProtoGetterExports);

const ClassInfo WebAssemblyInstancePrototype::s_info = { "WebAssembly.Instance"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(WebAssemblyInstancePrototype) };

static ALWAYS_INLINE JSWebAssemblyInstance* getInstance(JSGlobalObject* globalObject, VM& vm, JSValue v)
{
    auto throwScope = DECLARE_THROW_SCOPE(vm);
    JSWebAssemblyInstance* result = jsDynamicCast<JSWebAssemblyInstance*>(v);
    if (!result) {
        throwException(globalObject, throwScope, 
            createTypeError(globalObject, "expected |this| value to be an instance of WebAssembly.Instance"_s));
        return nullptr;
    }
    return result;
}

JSC_DEFINE_HOST_FUNCTION(webAssemblyInstanceProtoGetterExports, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);
    JSWebAssemblyInstance* instance = getInstance(globalObject, vm, callFrame->thisValue());
    RETURN_IF_EXCEPTION(scope, { });
    RELEASE_AND_RETURN(scope, JSValue::encode(instance->moduleRecord()->exportsObject()));
}

WebAssemblyInstancePrototype* WebAssemblyInstancePrototype::create(VM& vm, JSGlobalObject* globalObject, Structure* structure)
{
    auto* object = new (NotNull, allocateCell<WebAssemblyInstancePrototype>(vm)) WebAssemblyInstancePrototype(vm, structure);
    object->finishCreation(vm, globalObject);
    return object;
}

Structure* WebAssemblyInstancePrototype::createStructure(VM& vm, JSGlobalObject* globalObject, JSValue prototype)
{
    return Structure::create(vm, globalObject, prototype, TypeInfo(ObjectType, StructureFlags), info());
}

void WebAssemblyInstancePrototype::finishCreation(VM& vm, JSGlobalObject* globalObject)
{
    Base::finishCreation(vm);
    ASSERT(inherits(info()));
    JSC_TO_STRING_TAG_WITHOUT_TRANSITION();
    JSC_NATIVE_INTRINSIC_GETTER_WITHOUT_TRANSITION(vm.propertyNames->exports, webAssemblyInstanceProtoGetterExports, PropertyAttribute::ReadOnly, WebAssemblyInstanceExportsIntrinsic);
}

WebAssemblyInstancePrototype::WebAssemblyInstancePrototype(VM& vm, Structure* structure)
    : Base(vm, structure)
{
}

} // namespace JSC

#endif // ENABLE(WEBASSEMBLY)
