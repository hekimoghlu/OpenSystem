/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 16, 2024.
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
#include "WebAssemblyGCObjectBase.h"

#if ENABLE(WEBASSEMBLY)

#include "JSCInlines.h"
#include "TypeError.h"

namespace JSC {

const ClassInfo WebAssemblyGCObjectBase::s_info = { "WebAssemblyGCObjectBase"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(WebAssemblyGCObjectBase) };

WebAssemblyGCObjectBase::WebAssemblyGCObjectBase(VM& vm, Structure* structure, RefPtr<const Wasm::RTT>&& rtt)
    : Base(vm, structure)
    , m_rtt(WTFMove(rtt))
{
}

template<typename Visitor>
void WebAssemblyGCObjectBase::visitChildrenImpl(JSCell* cell, Visitor& visitor)
{
    WebAssemblyGCObjectBase* thisObject = jsCast<WebAssemblyGCObjectBase*>(cell);
    ASSERT_GC_OBJECT_INHERITS(thisObject, info());
    Base::visitChildren(thisObject, visitor);
}

DEFINE_VISIT_CHILDREN(WebAssemblyGCObjectBase);

bool WebAssemblyGCObjectBase::getOwnPropertySlot(JSObject* object, JSGlobalObject*, PropertyName, PropertySlot& slot)
{
    slot.setValue(object, static_cast<unsigned>(JSC::PropertyAttribute::None), jsUndefined());
    return false;
}

bool WebAssemblyGCObjectBase::getOwnPropertySlotByIndex(JSObject* object, JSGlobalObject*, unsigned, PropertySlot& slot)
{
    slot.setValue(object, static_cast<unsigned>(JSC::PropertyAttribute::None), jsUndefined());
    return false;
}

bool WebAssemblyGCObjectBase::put(JSCell*, JSGlobalObject* globalObject, PropertyName, JSValue, PutPropertySlot&)
{
    auto scope = DECLARE_THROW_SCOPE(globalObject->vm());
    return typeError(globalObject, scope, true, "Cannot set property for WebAssembly GC object"_s);
}

bool WebAssemblyGCObjectBase::putByIndex(JSCell*, JSGlobalObject* globalObject, unsigned, JSValue, bool)
{
    auto scope = DECLARE_THROW_SCOPE(globalObject->vm());
    return typeError(globalObject, scope, true, "Cannot set property for WebAssembly GC object"_s);
}

bool WebAssemblyGCObjectBase::deleteProperty(JSCell*, JSGlobalObject* globalObject, PropertyName, DeletePropertySlot&)
{
    auto scope = DECLARE_THROW_SCOPE(globalObject->vm());
    return typeError(globalObject, scope, true, "Cannot delete property for WebAssembly GC object"_s);
}

bool WebAssemblyGCObjectBase::deletePropertyByIndex(JSCell*, JSGlobalObject* globalObject, unsigned)
{
    auto scope = DECLARE_THROW_SCOPE(globalObject->vm());
    return typeError(globalObject, scope, true, "Cannot delete property for WebAssembly GC object"_s);
}

void WebAssemblyGCObjectBase::getOwnPropertyNames(JSObject*, JSGlobalObject*, PropertyNameArray& propertyNameArray, DontEnumPropertiesMode)
{
#if ASSERT_ENABLED
    ASSERT(!propertyNameArray.size());
#else
    UNUSED_PARAM(propertyNameArray);
#endif
    return;
}

bool WebAssemblyGCObjectBase::defineOwnProperty(JSObject*, JSGlobalObject* globalObject, PropertyName, const PropertyDescriptor&, bool shouldThrow)
{
    auto scope = DECLARE_THROW_SCOPE(globalObject->vm());
    return typeError(globalObject, scope, shouldThrow, "Cannot define property for WebAssembly GC object"_s);
}

JSValue WebAssemblyGCObjectBase::getPrototype(JSObject*, JSGlobalObject*)
{
    return jsNull();
}

bool WebAssemblyGCObjectBase::setPrototype(JSObject*, JSGlobalObject* globalObject, JSValue, bool shouldThrowIfCantSet)
{
    auto scope = DECLARE_THROW_SCOPE(globalObject->vm());
    return typeError(globalObject, scope, shouldThrowIfCantSet, "Cannot set prototype of WebAssembly GC object"_s);
}

bool WebAssemblyGCObjectBase::isExtensible(JSObject*, JSGlobalObject*)
{
    return false;
}

bool WebAssemblyGCObjectBase::preventExtensions(JSObject*, JSGlobalObject* globalObject)
{
    auto scope = DECLARE_THROW_SCOPE(globalObject->vm());
    return typeError(globalObject, scope, true, "Cannot run preventExtensions operation on WebAssembly GC object"_s);
}

} // namespace JSC

#endif // ENABLE(WEBASSEMBLY)
