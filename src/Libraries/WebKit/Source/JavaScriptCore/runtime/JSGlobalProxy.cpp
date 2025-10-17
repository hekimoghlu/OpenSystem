/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 8, 2022.
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
#include "JSGlobalProxy.h"

#include "JSGlobalObject.h"
#include "StructureInlines.h"

namespace JSC {

STATIC_ASSERT_IS_TRIVIALLY_DESTRUCTIBLE(JSGlobalProxy);

const ClassInfo JSGlobalProxy::s_info = { "JSGlobalProxy"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(JSGlobalProxy) };

template<typename Visitor>
void JSGlobalProxy::visitChildrenImpl(JSCell* cell, Visitor& visitor)
{
    JSGlobalProxy* thisObject = jsCast<JSGlobalProxy*>(cell);
    ASSERT_GC_OBJECT_INHERITS(thisObject, info());
    Base::visitChildren(thisObject, visitor);
    visitor.append(thisObject->m_target);
}

DEFINE_VISIT_CHILDREN_WITH_MODIFIER(JS_EXPORT_PRIVATE, JSGlobalProxy);

void JSGlobalProxy::setTarget(VM& vm, JSGlobalObject* globalObject)
{
    m_target.set(vm, this, globalObject);
    setPrototypeDirect(vm, globalObject->getPrototypeDirect());
    Structure* oldStructure = structure();
    DeferredStructureTransitionWatchpointFire deferred(vm, oldStructure);
    Structure* newStructure = Structure::changeGlobalProxyTargetTransition(vm, oldStructure, globalObject, deferred);
    setStructure(vm, newStructure);
}

bool JSGlobalProxy::getOwnPropertySlot(JSObject* object, JSGlobalObject* globalObject, PropertyName propertyName, PropertySlot& slot)
{
    JSGlobalProxy* thisObject = jsCast<JSGlobalProxy*>(object);
    return thisObject->target()->methodTable()->getOwnPropertySlot(thisObject->target(), globalObject, propertyName, slot);
}

bool JSGlobalProxy::getOwnPropertySlotByIndex(JSObject* object, JSGlobalObject* globalObject, unsigned propertyName, PropertySlot& slot)
{
    JSGlobalProxy* thisObject = jsCast<JSGlobalProxy*>(object);
    return thisObject->target()->methodTable()->getOwnPropertySlotByIndex(thisObject->target(), globalObject, propertyName, slot);
}

bool JSGlobalProxy::put(JSCell* cell, JSGlobalObject* globalObject, PropertyName propertyName, JSValue value, PutPropertySlot& slot)
{
    JSGlobalProxy* thisObject = jsCast<JSGlobalProxy*>(cell);
    return thisObject->target()->methodTable()->put(thisObject->target(), globalObject, propertyName, value, slot);
}

bool JSGlobalProxy::putByIndex(JSCell* cell, JSGlobalObject* globalObject, unsigned propertyName, JSValue value, bool shouldThrow)
{
    JSGlobalProxy* thisObject = jsCast<JSGlobalProxy*>(cell);
    return thisObject->target()->methodTable()->putByIndex(thisObject->target(), globalObject, propertyName, value, shouldThrow);
}

bool JSGlobalProxy::defineOwnProperty(JSObject* object, JSGlobalObject* globalObject, PropertyName propertyName, const PropertyDescriptor& descriptor, bool shouldThrow)
{
    JSGlobalProxy* thisObject = jsCast<JSGlobalProxy*>(object);
    return thisObject->target()->methodTable()->defineOwnProperty(thisObject->target(), globalObject, propertyName, descriptor, shouldThrow);
}

bool JSGlobalProxy::deleteProperty(JSCell* cell, JSGlobalObject* globalObject, PropertyName propertyName, DeletePropertySlot& slot)
{
    JSGlobalProxy* thisObject = jsCast<JSGlobalProxy*>(cell);
    return thisObject->target()->methodTable()->deleteProperty(thisObject->target(), globalObject, propertyName, slot);
}

bool JSGlobalProxy::isExtensible(JSObject* object, JSGlobalObject* globalObject)
{
    JSGlobalProxy* thisObject = jsCast<JSGlobalProxy*>(object);
    return thisObject->target()->methodTable()->isExtensible(thisObject->target(), globalObject);
}

bool JSGlobalProxy::preventExtensions(JSObject* object, JSGlobalObject* globalObject)
{
    JSGlobalProxy* thisObject = jsCast<JSGlobalProxy*>(object);
    return thisObject->target()->methodTable()->preventExtensions(thisObject->target(), globalObject);
}

bool JSGlobalProxy::deletePropertyByIndex(JSCell* cell, JSGlobalObject* globalObject, unsigned propertyName)
{
    JSGlobalProxy* thisObject = jsCast<JSGlobalProxy*>(cell);
    return thisObject->target()->methodTable()->deletePropertyByIndex(thisObject->target(), globalObject, propertyName);
}

void JSGlobalProxy::getOwnPropertyNames(JSObject* object, JSGlobalObject* globalObject, PropertyNameArray& propertyNames, DontEnumPropertiesMode mode)
{
    JSGlobalProxy* thisObject = jsCast<JSGlobalProxy*>(object);
    thisObject->target()->methodTable()->getOwnPropertyNames(thisObject->target(), globalObject, propertyNames, mode);
}

bool JSGlobalProxy::setPrototype(JSObject* object, JSGlobalObject* globalObject, JSValue prototype, bool shouldThrowIfCantSet)
{
    JSGlobalProxy* thisObject = jsCast<JSGlobalProxy*>(object);
    return thisObject->target()->methodTable()->setPrototype(thisObject->target(), globalObject, prototype, shouldThrowIfCantSet);
}

JSValue JSGlobalProxy::getPrototype(JSObject* object, JSGlobalObject* globalObject)
{
    JSGlobalProxy* thisObject = jsCast<JSGlobalProxy*>(object);
    return thisObject->target()->methodTable()->getPrototype(thisObject->target(), globalObject);
}

} // namespace JSC
