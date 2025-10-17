/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 14, 2022.
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
#include "JSAPIWrapperGlobalObject.h"

#include "JSCInlines.h"
#include "JSCallbackObject.h"
#include "Structure.h"
#include <wtf/NeverDestroyed.h>

class JSAPIWrapperGlobalObjectHandleOwner final : public JSC::WeakHandleOwner {
public:
    void finalize(JSC::Handle<JSC::Unknown>, void*) final;
};

static JSAPIWrapperGlobalObjectHandleOwner* jsAPIWrapperGlobalObjectHandleOwner()
{
    static NeverDestroyed<JSAPIWrapperGlobalObjectHandleOwner> jsWrapperGlobalObjectHandleOwner;
    return &jsWrapperGlobalObjectHandleOwner.get();
}

void JSAPIWrapperGlobalObjectHandleOwner::finalize(JSC::Handle<JSC::Unknown> handle, void*)
{
    auto* wrapperObject = static_cast<JSC::JSAPIWrapperGlobalObject*>(handle.get().asCell());
    if (!wrapperObject->wrappedObject())
        return;

    JSC::Heap::heap(wrapperObject)->releaseSoon(std::unique_ptr<JSC::JSCGLibWrapperObject>(wrapperObject->wrappedObject()));
    JSC::WeakSet::deallocate(JSC::WeakImpl::asWeakImpl(handle.slot()));
}

namespace JSC {

static JSC_DECLARE_HOST_FUNCTION(callJSAPIWrapperGlobalObjectCallbackObject);
static JSC_DECLARE_HOST_FUNCTION(constructJSAPIWrapperGlobalObjectCallbackObject);
static JSC_DECLARE_CUSTOM_GETTER(callbackGetterJSAPIWrapperGlobalObjectCallbackObject);
static JSC_DECLARE_CUSTOM_GETTER(staticFunctionGetterJSAPIWrapperGlobalObjectCallbackObject);

DEFINE_VISIT_CHILDREN_WITH_MODIFIER(template<>, JSCallbackObject<JSAPIWrapperGlobalObject>);

template <> const ClassInfo JSCallbackObject<JSAPIWrapperGlobalObject>::s_info = { "JSAPIWrapperGlobalObject"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(JSCallbackObject) };
template<> const DestructionMode JSCallbackObject<JSAPIWrapperGlobalObject>::needsDestruction = NeedsDestruction;

template <>
NativeFunction::Ptr JSCallbackObject<JSAPIWrapperGlobalObject>::getCallFunction()
{
    return callJSAPIWrapperGlobalObjectCallbackObject;
}

template <>
NativeFunction::Ptr JSCallbackObject<JSAPIWrapperGlobalObject>::getConstructFunction()
{
    return constructJSAPIWrapperGlobalObjectCallbackObject;
}

template <>
GetValueFunc JSCallbackObject<JSAPIWrapperGlobalObject>::getCallbackGetter()
{
    return callbackGetterJSAPIWrapperGlobalObjectCallbackObject;
}

template <>
GetValueFunc JSCallbackObject<JSAPIWrapperGlobalObject>::getStaticFunctionGetter()
{
    return staticFunctionGetterJSAPIWrapperGlobalObjectCallbackObject;
}

template <>
GCClient::IsoSubspace* JSCallbackObject<JSAPIWrapperGlobalObject>::subspaceForImpl(VM& vm, SubspaceAccess mode)
{
    switch (mode) {
    case SubspaceAccess::OnMainThread:
        return vm.callbackAPIWrapperGlobalObjectSpace<SubspaceAccess::OnMainThread>();
    case SubspaceAccess::Concurrently:
        return vm.callbackAPIWrapperGlobalObjectSpace<SubspaceAccess::Concurrently>();
    }
    RELEASE_ASSERT_NOT_REACHED();
    return nullptr;
}

template <>
Structure* JSCallbackObject<JSAPIWrapperGlobalObject>::createStructure(VM& vm, JSGlobalObject*, JSValue proto)
{
    return Structure::create(vm, nullptr, proto, TypeInfo(GlobalObjectType, StructureFlags), &s_info);
}

template<>
JSCallbackObject<JSAPIWrapperGlobalObject>* JSCallbackObject<JSAPIWrapperGlobalObject>::create(VM& vm, JSClassRef classRef, Structure* structure)
{
    JSCallbackObject<JSAPIWrapperGlobalObject>* callbackObject = new (NotNull, allocateCell<JSCallbackObject<JSAPIWrapperGlobalObject>>(vm)) JSCallbackObject(vm, classRef, structure);
    callbackObject->finishCreation(vm);
    return callbackObject;
}

JSC_DEFINE_HOST_FUNCTION(callJSAPIWrapperGlobalObjectCallbackObject, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    return JSCallbackObject<JSAPIWrapperGlobalObject>::callImpl(globalObject, callFrame);
}

JSC_DEFINE_HOST_FUNCTION(constructJSAPIWrapperGlobalObjectCallbackObject, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    return JSCallbackObject<JSAPIWrapperGlobalObject>::constructImpl(globalObject, callFrame);
}

JSC_DEFINE_CUSTOM_GETTER(callbackGetterJSAPIWrapperGlobalObjectCallbackObject, (JSGlobalObject* globalObject, EncodedJSValue thisValue, PropertyName propertyName))
{
    return JSCallbackObject<JSAPIWrapperGlobalObject>::callbackGetterImpl(globalObject, thisValue, propertyName);
}

JSC_DEFINE_CUSTOM_GETTER(staticFunctionGetterJSAPIWrapperGlobalObjectCallbackObject, (JSGlobalObject* globalObject, EncodedJSValue thisValue, PropertyName propertyName))
{
    return JSCallbackObject<JSAPIWrapperGlobalObject>::staticFunctionGetterImpl(globalObject, thisValue, propertyName);
}

JSAPIWrapperGlobalObject::JSAPIWrapperGlobalObject(VM& vm, Structure* structure)
    : Base(vm, structure)
{
}

void JSAPIWrapperGlobalObject::finishCreation(VM& vm)
{
    Base::finishCreation(vm);
    WeakSet::allocate(this, jsAPIWrapperGlobalObjectHandleOwner(), 0); // Balanced in JSAPIWrapperGlobalObjectHandleOwner::finalize.
}

} // namespace JSC
