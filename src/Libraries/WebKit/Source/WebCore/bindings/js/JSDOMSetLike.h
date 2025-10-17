/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 19, 2022.
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
#pragma once

#include "JSDOMBinding.h"
#include "JSDOMConvert.h"
#include "JSDOMGlobalObject.h"
#include <JavaScriptCore/BuiltinNames.h>
#include <JavaScriptCore/CommonIdentifiers.h>

namespace WebCore {

// FIXME: Optimize / rework maplike<> and setlike<> declarations.
// A few ideas in https://bugs.webkit.org/show_bug.cgi?id=241639.

WEBCORE_EXPORT std::pair<bool, std::reference_wrapper<JSC::JSObject>> getBackingSet(JSC::JSGlobalObject&, JSC::JSObject& setLike);
WEBCORE_EXPORT JSC::JSValue forwardForEachCallToBackingSet(JSDOMGlobalObject&, JSC::CallFrame&, JSC::JSObject& setLike);
WEBCORE_EXPORT JSC::JSValue forwardAttributeGetterToBackingSet(JSC::JSGlobalObject&, JSC::JSObject&, const JSC::Identifier&);
WEBCORE_EXPORT JSC::JSValue forwardFunctionCallToBackingSet(JSC::JSGlobalObject&, JSC::CallFrame&, JSC::JSObject&, const JSC::Identifier&);
WEBCORE_EXPORT void clearBackingSet(JSC::JSGlobalObject&, JSC::JSObject&);
WEBCORE_EXPORT void addToBackingSet(JSC::JSGlobalObject&, JSC::JSObject&, JSC::JSValue item);

template<typename WrapperClass> JSC::JSObject& getAndInitializeBackingSet(JSC::JSGlobalObject&, WrapperClass&);
template<typename WrapperClass> JSC::JSValue forwardSizeToSetLike(JSC::JSGlobalObject&, WrapperClass&);
template<typename WrapperClass> JSC::JSValue forwardEntriesToSetLike(JSC::JSGlobalObject&, JSC::CallFrame&, WrapperClass&);
template<typename WrapperClass> JSC::JSValue forwardKeysToSetLike(JSC::JSGlobalObject&, JSC::CallFrame&, WrapperClass&);
template<typename WrapperClass> JSC::JSValue forwardValuesToSetLike(JSC::JSGlobalObject&, JSC::CallFrame&, WrapperClass&);
template<typename WrapperClass, typename Callback> JSC::JSValue forwardForEachToSetLike(JSC::JSGlobalObject&, JSC::CallFrame&, WrapperClass&, Callback&&);
template<typename WrapperClass> JSC::JSValue forwardClearToSetLike(JSC::JSGlobalObject&, JSC::CallFrame&, WrapperClass&);

template<typename WrapperClass, typename ItemType> JSC::JSValue forwardHasToSetLike(JSC::JSGlobalObject&, JSC::CallFrame&, WrapperClass&, ItemType&&);
template<typename WrapperClass, typename ItemType> JSC::JSValue forwardAddToSetLike(JSC::JSGlobalObject&, JSC::CallFrame&, WrapperClass&, ItemType&&);
template<typename WrapperClass, typename ItemType> JSC::JSValue forwardDeleteToSetLike(JSC::JSGlobalObject&, JSC::CallFrame&, WrapperClass&, ItemType&&);

class DOMSetAdapter {
public:
    DOMSetAdapter(JSC::JSGlobalObject&, JSC::JSObject&);
    template<typename IDLType> void add(typename IDLType::ParameterType value);
    void clear();

private:
    JSC::JSGlobalObject& m_lexicalGlobalObject;
    JSC::JSObject& m_backingSet;
};

inline DOMSetAdapter::DOMSetAdapter(JSC::JSGlobalObject& lexicalGlobalObject, JSC::JSObject& backingSet)
    : m_lexicalGlobalObject(lexicalGlobalObject)
    , m_backingSet(backingSet)
{
}

template<typename IDLType>
void DOMSetAdapter::add(typename IDLType::ParameterType value)
{
    JSC::JSLockHolder locker(&m_lexicalGlobalObject);
    auto item = toJS<IDLType>(m_lexicalGlobalObject, *JSC::jsCast<JSDOMGlobalObject*>(&m_lexicalGlobalObject), std::forward<typename IDLType::ParameterType>(value));
    addToBackingSet(m_lexicalGlobalObject, m_backingSet, item);
}

template<typename WrapperClass>
JSC::JSObject& getAndInitializeBackingSet(JSC::JSGlobalObject& lexicalGlobalObject, WrapperClass& setLike)
{
    auto pair = getBackingSet(lexicalGlobalObject, setLike);
    if (pair.first) {
        DOMSetAdapter adapter { lexicalGlobalObject, pair.second.get() };
        setLike.wrapped().initializeSetLike(adapter);
    }
    return pair.second.get();
}

template<typename WrapperClass>
JSC::JSValue forwardSizeToSetLike(JSC::JSGlobalObject& lexicalGlobalObject, WrapperClass& setLike)
{
    auto& vm = JSC::getVM(&lexicalGlobalObject);
    return forwardAttributeGetterToBackingSet(lexicalGlobalObject, getAndInitializeBackingSet(lexicalGlobalObject, setLike), vm.propertyNames->builtinNames().sizePrivateName());
}

template<typename WrapperClass>
JSC::JSValue forwardEntriesToSetLike(JSC::JSGlobalObject& lexicalGlobalObject, JSC::CallFrame& callFrame, WrapperClass& setLike)
{
    auto& vm = JSC::getVM(&lexicalGlobalObject);
    return forwardFunctionCallToBackingSet(lexicalGlobalObject, callFrame, getAndInitializeBackingSet(lexicalGlobalObject, setLike), vm.propertyNames->builtinNames().entriesPrivateName());
}

template<typename WrapperClass>
JSC::JSValue forwardKeysToSetLike(JSC::JSGlobalObject& lexicalGlobalObject, JSC::CallFrame& callFrame, WrapperClass& setLike)
{
    auto& vm = JSC::getVM(&lexicalGlobalObject);
    return forwardFunctionCallToBackingSet(lexicalGlobalObject, callFrame, getAndInitializeBackingSet(lexicalGlobalObject, setLike), vm.propertyNames->builtinNames().keysPrivateName());
}

template<typename WrapperClass>
JSC::JSValue forwardValuesToSetLike(JSC::JSGlobalObject& lexicalGlobalObject, JSC::CallFrame& callFrame, WrapperClass& setLike)
{
    auto& vm = JSC::getVM(&lexicalGlobalObject);
    return forwardFunctionCallToBackingSet(lexicalGlobalObject, callFrame, getAndInitializeBackingSet(lexicalGlobalObject, setLike), vm.propertyNames->builtinNames().valuesPrivateName());
}

template<typename WrapperClass, typename Callback>
JSC::JSValue forwardForEachToSetLike(JSC::JSGlobalObject& lexicalGlobalObject, JSC::CallFrame& callFrame, WrapperClass& setLike, Callback&&)
{
    getAndInitializeBackingSet(lexicalGlobalObject, setLike);
    return forwardForEachCallToBackingSet(*JSC::jsCast<JSDOMGlobalObject*>(&lexicalGlobalObject), callFrame, setLike);
}

template<typename WrapperClass>
JSC::JSValue forwardClearToSetLike(JSC::JSGlobalObject& lexicalGlobalObject, JSC::CallFrame& callFrame, WrapperClass& setLike)
{
    setLike.wrapped().clearFromSetLike();

    auto& vm = JSC::getVM(&lexicalGlobalObject);
    return forwardFunctionCallToBackingSet(lexicalGlobalObject, callFrame, getAndInitializeBackingSet(lexicalGlobalObject, setLike), vm.propertyNames->builtinNames().clearPrivateName());
}

template<typename WrapperClass, typename ItemType>
JSC::JSValue forwardHasToSetLike(JSC::JSGlobalObject& lexicalGlobalObject, JSC::CallFrame& callFrame, WrapperClass& setLike, ItemType&&)
{
    auto& vm = JSC::getVM(&lexicalGlobalObject);
    return forwardFunctionCallToBackingSet(lexicalGlobalObject, callFrame, getAndInitializeBackingSet(lexicalGlobalObject, setLike), vm.propertyNames->builtinNames().hasPrivateName());
}

template<typename WrapperClass, typename ItemType>
JSC::JSValue forwardAddToSetLike(JSC::JSGlobalObject& lexicalGlobalObject, JSC::CallFrame& callFrame, WrapperClass& setLike, ItemType&& item)
{
    setLike.wrapped().addToSetLike(std::forward<ItemType>(item));

    auto& vm = JSC::getVM(&lexicalGlobalObject);
    forwardFunctionCallToBackingSet(lexicalGlobalObject, callFrame, getAndInitializeBackingSet(lexicalGlobalObject, setLike), vm.propertyNames->builtinNames().addPrivateName());
    return &setLike;
}

template<typename WrapperClass, typename ItemType>
JSC::JSValue forwardDeleteToSetLike(JSC::JSGlobalObject& lexicalGlobalObject, JSC::CallFrame& callFrame, WrapperClass& setLike, ItemType&& item)
{
    // Initialize backingSet before removing value so assertion below actually holds.
    auto& backingSet = getAndInitializeBackingSet(lexicalGlobalObject, setLike);

    auto isDeleted = setLike.wrapped().removeFromSetLike(std::forward<ItemType>(item));
    UNUSED_PARAM(isDeleted);

    auto& vm = JSC::getVM(&lexicalGlobalObject);
    auto result = forwardFunctionCallToBackingSet(lexicalGlobalObject, callFrame, backingSet, vm.propertyNames->builtinNames().deletePrivateName());
    ASSERT_UNUSED(result, result.asBoolean() == isDeleted);
    return result;
}

}
