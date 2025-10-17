/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 1, 2025.
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
#include "JSCallbackObject.h"

#include "JSCInlines.h"

namespace JSC {

static JSC_DECLARE_HOST_FUNCTION(callJSNonFinalObjectCallbackObject);
static JSC_DECLARE_HOST_FUNCTION(constructJSNonFinalObjectCallbackObject);
static JSC_DECLARE_HOST_FUNCTION(callJSGlobalObjectCallbackObject);
static JSC_DECLARE_HOST_FUNCTION(constructJSGlobalObjectCallbackObject);

static JSC_DECLARE_CUSTOM_GETTER(callbackGetterJSNonFinalObjectCallbackObject);
static JSC_DECLARE_CUSTOM_GETTER(staticFunctionGetterJSNonFinalObjectCallbackObject);
static JSC_DECLARE_CUSTOM_GETTER(callbackGetterJSGlobalObjectCallbackObject);
static JSC_DECLARE_CUSTOM_GETTER(staticFunctionGetterJSGlobalObjectCallbackObject);

DEFINE_VISIT_CHILDREN_WITH_MODIFIER(template<>, JSCallbackObject<JSNonFinalObject>);
DEFINE_VISIT_CHILDREN_WITH_MODIFIER(template<>, JSCallbackObject<JSGlobalObject>);

// Define the two types of JSCallbackObjects we support.
template <> const ClassInfo JSCallbackObject<JSNonFinalObject>::s_info = { "CallbackObject"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(JSCallbackObject) };
template <> const ClassInfo JSCallbackObject<JSGlobalObject>::s_info = { "CallbackGlobalObject"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(JSCallbackObject) };
template<> const DestructionMode JSCallbackObject<JSNonFinalObject>::needsDestruction = NeedsDestruction;
template<> const DestructionMode JSCallbackObject<JSGlobalObject>::needsDestruction = NeedsDestruction;

template<>
NativeFunction::Ptr JSCallbackObject<JSNonFinalObject>::getCallFunction()
{
    return callJSNonFinalObjectCallbackObject;
}

template<>
NativeFunction::Ptr JSCallbackObject<JSNonFinalObject>::getConstructFunction()
{
    return constructJSNonFinalObjectCallbackObject;
}

template <>
GetValueFunc JSCallbackObject<JSNonFinalObject>::getCallbackGetter()
{
    return callbackGetterJSNonFinalObjectCallbackObject;
}

template <>
GetValueFunc JSCallbackObject<JSNonFinalObject>::getStaticFunctionGetter()
{
    return staticFunctionGetterJSNonFinalObjectCallbackObject;
}


template<>
NativeFunction::Ptr JSCallbackObject<JSGlobalObject>::getCallFunction()
{
    return callJSGlobalObjectCallbackObject;
}

template<>
NativeFunction::Ptr JSCallbackObject<JSGlobalObject>::getConstructFunction()
{
    return constructJSGlobalObjectCallbackObject;
}

template <>
GetValueFunc JSCallbackObject<JSGlobalObject>::getCallbackGetter()
{
    return callbackGetterJSGlobalObjectCallbackObject;
}

template <>
GetValueFunc JSCallbackObject<JSGlobalObject>::getStaticFunctionGetter()
{
    return staticFunctionGetterJSGlobalObjectCallbackObject;
}

template<>
JSCallbackObject<JSGlobalObject>* JSCallbackObject<JSGlobalObject>::create(VM& vm, JSClassRef classRef, Structure* structure)
{
    JSCallbackObject<JSGlobalObject>* callbackObject = new (NotNull, allocateCell<JSCallbackObject<JSGlobalObject>>(vm)) JSCallbackObject(vm, classRef, structure);
    callbackObject->finishCreation(vm);
    return callbackObject;
}

template <>
Structure* JSCallbackObject<JSNonFinalObject>::createStructure(VM& vm, JSGlobalObject* globalObject, JSValue proto)
{ 
    return Structure::create(vm, globalObject, proto, TypeInfo(ObjectType, StructureFlags), info()); 
}
    
template <>
Structure* JSCallbackObject<JSGlobalObject>::createStructure(VM& vm, JSGlobalObject* globalObject, JSValue proto)
{ 
    return Structure::create(vm, globalObject, proto, TypeInfo(GlobalObjectType, StructureFlags), info()); 
}

template <>
GCClient::IsoSubspace* JSCallbackObject<JSNonFinalObject>::subspaceForImpl(VM& vm, SubspaceAccess mode)
{
    switch (mode) {
    case SubspaceAccess::OnMainThread:
        return vm.callbackObjectSpace<SubspaceAccess::OnMainThread>();
    case SubspaceAccess::Concurrently:
        return vm.callbackObjectSpace<SubspaceAccess::Concurrently>();
    }
    RELEASE_ASSERT_NOT_REACHED();
    return nullptr;
}

template <>
GCClient::IsoSubspace* JSCallbackObject<JSGlobalObject>::subspaceForImpl(VM& vm, SubspaceAccess mode)
{
    switch (mode) {
    case SubspaceAccess::OnMainThread:
        return vm.callbackGlobalObjectSpace<SubspaceAccess::OnMainThread>();
    case SubspaceAccess::Concurrently:
        return vm.callbackGlobalObjectSpace<SubspaceAccess::Concurrently>();
    }
    RELEASE_ASSERT_NOT_REACHED();
    return nullptr;
}

JSC_DEFINE_HOST_FUNCTION(callJSNonFinalObjectCallbackObject, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    return JSCallbackObject<JSNonFinalObject>::callImpl(globalObject, callFrame);
}

JSC_DEFINE_HOST_FUNCTION(constructJSNonFinalObjectCallbackObject, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    return JSCallbackObject<JSNonFinalObject>::constructImpl(globalObject, callFrame);
}

JSC_DEFINE_HOST_FUNCTION(callJSGlobalObjectCallbackObject, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    return JSCallbackObject<JSGlobalObject>::callImpl(globalObject, callFrame);
}

JSC_DEFINE_HOST_FUNCTION(constructJSGlobalObjectCallbackObject, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    return JSCallbackObject<JSGlobalObject>::constructImpl(globalObject, callFrame);
}

JSC_DEFINE_CUSTOM_GETTER(callbackGetterJSNonFinalObjectCallbackObject, (JSGlobalObject* globalObject, EncodedJSValue thisValue, PropertyName propertyName))
{
    return JSCallbackObject<JSNonFinalObject>::callbackGetterImpl(globalObject, thisValue, propertyName);
}

JSC_DEFINE_CUSTOM_GETTER(staticFunctionGetterJSNonFinalObjectCallbackObject, (JSGlobalObject* globalObject, EncodedJSValue thisValue, PropertyName propertyName))
{
    return JSCallbackObject<JSNonFinalObject>::staticFunctionGetterImpl(globalObject, thisValue, propertyName);
}

JSC_DEFINE_CUSTOM_GETTER(callbackGetterJSGlobalObjectCallbackObject, (JSGlobalObject* globalObject, EncodedJSValue thisValue, PropertyName propertyName))
{
    return JSCallbackObject<JSGlobalObject>::callbackGetterImpl(globalObject, thisValue, propertyName);
}

JSC_DEFINE_CUSTOM_GETTER(staticFunctionGetterJSGlobalObjectCallbackObject, (JSGlobalObject* globalObject, EncodedJSValue thisValue, PropertyName propertyName))
{
    return JSCallbackObject<JSGlobalObject>::staticFunctionGetterImpl(globalObject, thisValue, propertyName);
}

} // namespace JSC
