/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 28, 2024.
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
#include <JavaScriptCore/BuiltinNames.h>
#include <JavaScriptCore/CommonIdentifiers.h>

namespace WebCore {

// FIXME: Optimize / rework maplike<> and setlike<> declarations.
// A few ideas in https://bugs.webkit.org/show_bug.cgi?id=241639.

WEBCORE_EXPORT std::pair<bool, std::reference_wrapper<JSC::JSObject>> getBackingMap(JSC::JSGlobalObject&, JSC::JSObject& mapLike);
WEBCORE_EXPORT JSC::JSValue forwardAttributeGetterToBackingMap(JSC::JSGlobalObject&, JSC::JSObject&, const JSC::Identifier&);
WEBCORE_EXPORT JSC::JSValue forwardFunctionCallToBackingMap(JSC::JSGlobalObject&, JSC::CallFrame&, JSC::JSObject&, const JSC::Identifier&);
WEBCORE_EXPORT JSC::JSValue forwardForEachCallToBackingMap(JSDOMGlobalObject&, JSC::CallFrame&, JSC::JSObject&);
WEBCORE_EXPORT void clearBackingMap(JSC::JSGlobalObject&, JSC::JSObject&);
WEBCORE_EXPORT void setToBackingMap(JSC::JSGlobalObject&, JSC::JSObject&, JSC::JSValue, JSC::JSValue);

template<typename WrapperClass> JSC::JSValue forwardSizeToMapLike(JSC::JSGlobalObject&, WrapperClass&);
template<typename WrapperClass> JSC::JSValue forwardEntriesToMapLike(JSC::JSGlobalObject&, JSC::CallFrame&, WrapperClass&);
template<typename WrapperClass> JSC::JSValue forwardKeysToMapLike(JSC::JSGlobalObject&, JSC::CallFrame&, WrapperClass&);
template<typename WrapperClass> JSC::JSValue forwardValuesToMapLike(JSC::JSGlobalObject&, JSC::CallFrame&, WrapperClass&);
template<typename WrapperClass> JSC::JSValue forwardClearToMapLike(JSC::JSGlobalObject&, JSC::CallFrame&, WrapperClass&);
template<typename WrapperClass, typename Callback> JSC::JSValue forwardForEachToMapLike(JSC::JSGlobalObject&, JSC::CallFrame&, WrapperClass&, Callback&&);
template<typename WrapperClass, typename ItemType> JSC::JSValue forwardGetToMapLike(JSC::JSGlobalObject&, JSC::CallFrame&, WrapperClass&, ItemType&&);
template<typename WrapperClass, typename ItemType> JSC::JSValue forwardHasToMapLike(JSC::JSGlobalObject&, JSC::CallFrame&, WrapperClass&, ItemType&&);
template<typename WrapperClass, typename ItemType, typename ValueType> JSC::JSValue forwardSetToMapLike(JSC::JSGlobalObject&, JSC::CallFrame&, WrapperClass&, ItemType&&, ValueType&&);
template<typename WrapperClass, typename ItemType> JSC::JSValue forwardDeleteToMapLike(JSC::JSGlobalObject&, JSC::CallFrame&, WrapperClass&, ItemType&&);

class DOMMapAdapter {
public:
    DOMMapAdapter(JSC::JSGlobalObject&, JSC::JSObject&);
    template<typename IDLKeyType, typename IDLValueType> void set(typename IDLKeyType::ParameterType, typename IDLValueType::ParameterType);
    void clear();

private:
    JSC::JSGlobalObject& m_lexicalGlobalObject;
    JSC::JSObject& m_backingMap;
};

inline DOMMapAdapter::DOMMapAdapter(JSC::JSGlobalObject& lexicalGlobalObject, JSC::JSObject& backingMap)
    : m_lexicalGlobalObject(lexicalGlobalObject)
    , m_backingMap(backingMap)
{
}

template<typename IDLKeyType, typename IDLValueType>
void DOMMapAdapter::set(typename IDLKeyType::ParameterType key, typename IDLValueType::ParameterType value)
{
    JSC::JSLockHolder locker(&m_lexicalGlobalObject);
    auto jsKey = toJS<IDLKeyType>(m_lexicalGlobalObject, *JSC::jsCast<JSDOMGlobalObject*>(&m_lexicalGlobalObject), std::forward<typename IDLKeyType::ParameterType>(key));
    auto jsValue = toJS<IDLValueType>(m_lexicalGlobalObject, *JSC::jsCast<JSDOMGlobalObject*>(&m_lexicalGlobalObject), std::forward<typename IDLValueType::ParameterType>(value));
    setToBackingMap(m_lexicalGlobalObject, m_backingMap, jsKey, jsValue);
}

template<typename WrapperClass> JSC::JSObject& getAndInitializeBackingMap(JSC::JSGlobalObject& lexicalGlobalObject, WrapperClass& mapLike)
{
    auto pair = getBackingMap(lexicalGlobalObject, mapLike);
    if (pair.first) {
        DOMMapAdapter adapter { lexicalGlobalObject, pair.second.get() };
        mapLike.wrapped().initializeMapLike(adapter);
    }
    return pair.second.get();
}

template<typename WrapperClass> JSC::JSValue forwardSizeToMapLike(JSC::JSGlobalObject& lexicalGlobalObject, WrapperClass& mapLike)
{
    auto& vm = JSC::getVM(&lexicalGlobalObject);
    return forwardAttributeGetterToBackingMap(lexicalGlobalObject, getAndInitializeBackingMap(lexicalGlobalObject, mapLike), vm.propertyNames->builtinNames().sizePrivateName());
}

template<typename WrapperClass> JSC::JSValue forwardEntriesToMapLike(JSC::JSGlobalObject& lexicalGlobalObject, JSC::CallFrame& callFrame, WrapperClass& mapLike)
{
    auto& vm = JSC::getVM(&lexicalGlobalObject);
    return forwardFunctionCallToBackingMap(lexicalGlobalObject, callFrame, getAndInitializeBackingMap(lexicalGlobalObject, mapLike), vm.propertyNames->builtinNames().entriesPrivateName());
}

template<typename WrapperClass> JSC::JSValue forwardKeysToMapLike(JSC::JSGlobalObject& lexicalGlobalObject, JSC::CallFrame& callFrame, WrapperClass& mapLike)
{
    auto& vm = JSC::getVM(&lexicalGlobalObject);
    return forwardFunctionCallToBackingMap(lexicalGlobalObject, callFrame, getAndInitializeBackingMap(lexicalGlobalObject, mapLike), vm.propertyNames->builtinNames().keysPrivateName());
}

template<typename WrapperClass> JSC::JSValue forwardValuesToMapLike(JSC::JSGlobalObject& lexicalGlobalObject, JSC::CallFrame& callFrame, WrapperClass& mapLike)
{
    auto& vm = JSC::getVM(&lexicalGlobalObject);
    return forwardFunctionCallToBackingMap(lexicalGlobalObject, callFrame, getAndInitializeBackingMap(lexicalGlobalObject, mapLike), vm.propertyNames->builtinNames().valuesPrivateName());
}

template<typename WrapperClass> JSC::JSValue forwardClearToMapLike(JSC::JSGlobalObject& lexicalGlobalObject, JSC::CallFrame& callFrame, WrapperClass& mapLike)
{
    mapLike.wrapped().clear();
    auto& vm = JSC::getVM(&lexicalGlobalObject);
    return forwardFunctionCallToBackingMap(lexicalGlobalObject, callFrame, getAndInitializeBackingMap(lexicalGlobalObject, mapLike), vm.propertyNames->builtinNames().clearPrivateName());
}

template<typename WrapperClass, typename Callback> JSC::JSValue forwardForEachToMapLike(JSC::JSGlobalObject& lexicalGlobalObject, JSC::CallFrame& callFrame, WrapperClass& mapLike, Callback&&)
{
    getAndInitializeBackingMap(lexicalGlobalObject, mapLike);
    return forwardForEachCallToBackingMap(*JSC::jsCast<JSDOMGlobalObject*>(&lexicalGlobalObject), callFrame, mapLike);
}

template<typename WrapperClass, typename ItemType> JSC::JSValue forwardGetToMapLike(JSC::JSGlobalObject& lexicalGlobalObject, JSC::CallFrame& callFrame, WrapperClass& mapLike, ItemType&&)
{
    auto& vm = JSC::getVM(&lexicalGlobalObject);
    return forwardFunctionCallToBackingMap(lexicalGlobalObject, callFrame, getAndInitializeBackingMap(lexicalGlobalObject, mapLike), vm.propertyNames->builtinNames().getPrivateName());
}

template<typename WrapperClass, typename ItemType> JSC::JSValue forwardHasToMapLike(JSC::JSGlobalObject& lexicalGlobalObject, JSC::CallFrame& callFrame, WrapperClass& mapLike, ItemType&&)
{
    auto& vm = JSC::getVM(&lexicalGlobalObject);
    return forwardFunctionCallToBackingMap(lexicalGlobalObject, callFrame, getAndInitializeBackingMap(lexicalGlobalObject, mapLike), vm.propertyNames->builtinNames().hasPrivateName());
}

template<typename WrapperClass, typename KeyType, typename ValueType> JSC::JSValue forwardSetToMapLike(JSC::JSGlobalObject& lexicalGlobalObject, JSC::CallFrame& callFrame, WrapperClass& mapLike, KeyType&& key, ValueType&& value)
{
    auto& vm = JSC::getVM(&lexicalGlobalObject);
    mapLike.wrapped().setFromMapLike(std::forward<KeyType>(key), std::forward<ValueType>(value));
    forwardFunctionCallToBackingMap(lexicalGlobalObject, callFrame, getAndInitializeBackingMap(lexicalGlobalObject, mapLike), vm.propertyNames->builtinNames().setPrivateName());
    return &mapLike;
}

template<typename WrapperClass, typename ItemType> JSC::JSValue forwardDeleteToMapLike(JSC::JSGlobalObject& lexicalGlobalObject, JSC::CallFrame& callFrame, WrapperClass& mapLike, ItemType&& item)
{
    auto& vm = JSC::getVM(&lexicalGlobalObject);
    auto& backingMap = getAndInitializeBackingMap(lexicalGlobalObject, mapLike);

    auto isDeleted = mapLike.wrapped().remove(std::forward<ItemType>(item));
    UNUSED_PARAM(isDeleted);

    auto result = forwardFunctionCallToBackingMap(lexicalGlobalObject, callFrame, backingMap, vm.propertyNames->builtinNames().deletePrivateName());

    ASSERT_UNUSED(result, result.asBoolean() == isDeleted);
    return result;
}

}
