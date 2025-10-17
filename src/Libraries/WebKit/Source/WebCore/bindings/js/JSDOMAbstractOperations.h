/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 18, 2025.
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

#include "JSDOMConvertStrings.h"
#include "JSDOMExceptionHandling.h"

namespace WebCore {

// Implementations of the abstract operations defined at
// https://webidl.spec.whatwg.org/#legacy-platform-object-abstract-ops

enum class LegacyOverrideBuiltIns : bool { No, Yes };

// An implementation of the 'named property visibility algorithm'
// https://webidl.spec.whatwg.org/#dfn-named-property-visibility
template<LegacyOverrideBuiltIns overrideBuiltins, class JSClass>
static bool isVisibleNamedProperty(JSC::JSGlobalObject& lexicalGlobalObject, JSClass& thisObject, JSC::PropertyName propertyName)
{
    // FIXME: It seems unfortunate that have to do two lookups for the property name,
    // one for isSupportedPropertyName and one by the user of this algorithm to access
    // that property. It would be nice if we could smuggle the result, or an iterator
    // out so the duplicate lookup could be avoided.

    // NOTE: While it is not specified, a Symbol can never be a 'supported property
    // name' so we check that first.
    if (propertyName.isSymbol())
        return false;

    auto& impl = thisObject.wrapped();

    // 1. If P is not a supported property name of O, then return false.
    if (!impl.isSupportedPropertyName(propertyNameToString(propertyName)))
        return false;
    
    // 2. If O has an own property named P, then return false.
    JSC::PropertySlot slot { &thisObject, JSC::PropertySlot::InternalMethodType::VMInquiry, &lexicalGlobalObject.vm() };
    if (JSC::JSObject::getOwnPropertySlot(&thisObject, &lexicalGlobalObject, propertyName, slot))
        return false;
    
    // 3. If O implements an interface that has the [LegacyOverrideBuiltIns] extended attribute, then return true.
    if (overrideBuiltins == LegacyOverrideBuiltIns::Yes)
        return true;

    // 4. Initialize prototype to be the value of the internal [[Prototype]] property of O.
    // 5. While prototype is not null:
    //    1. If prototype is not a named properties object, and prototype has an own property named P, then return false.
    // FIXME: Implement checking for 'named properties object'.
    //    2. Set prototype to be the value of the internal [[Prototype]] property of prototype.
    auto prototype = thisObject.getPrototypeDirect();
    if (prototype.isObject() && JSC::asObject(prototype)->getPropertySlot(&lexicalGlobalObject, propertyName, slot))
        return false;

    // 6. Return true.
    return true;
}

// NOTE: Named getters are little odd. To avoid doing duplicate lookups (once when checking if
//       the property name is a 'supported property name' and once to get the value) we signal
//       that a property is supported by whether or not it is 'null' (where what null means is
//       dependant on the IDL type). This is based on the assumption that no named getter will
//       ever actually want to return null as an actual return value, which seems like an ok
//       assumption to make (should it turn out this doesn't hold in the future, we have lots
//       of options; do two lookups, add an extra layer of Optional, etc.).
template<typename IDLType, typename JSClass, typename InnerItemAccessor>
static decltype(auto) visibleNamedPropertyItemAccessorFunctor(InnerItemAccessor&& innerItemAccessor)
{
    if constexpr (IsExceptionOr<std::invoke_result_t<InnerItemAccessor, JSClass&, JSC::PropertyName>>) {
        using ReturnType = ExceptionOr<typename IDLType::ImplementationType>;

        return [innerItemAccessor = std::forward<InnerItemAccessor>(innerItemAccessor)] (JSClass& thisObject, JSC::PropertyName propertyName) -> std::optional<ReturnType> {
            auto result = innerItemAccessor(thisObject, propertyName);
            if (result.hasException())
                return ReturnType { result.releaseException() };
            if (!IDLType::isNullValue(result.returnValue()))
                return ReturnType { IDLType::extractValueFromNullable(result.releaseReturnValue()) };
            return std::nullopt;
        };
    } else {
        using ReturnType = typename IDLType::ImplementationType;

        return [innerItemAccessor = std::forward<InnerItemAccessor>(innerItemAccessor)] (JSClass& thisObject, JSC::PropertyName propertyName) -> std::optional<ReturnType> {
            auto result = innerItemAccessor(thisObject, propertyName);
            if (!IDLType::isNullValue(result))
                return ReturnType { IDLType::extractValueFromNullable(result) };
            return std::nullopt;
        };
    }
}

// An implementation of the 'named property visibility algorithm' augmented to replace the
// 'supported property name' check with direct access to the implementation value returned
// for the property name, via passed in functor. This allows us to avoid two looking up the
// the property name twice; once for 'named property visibility algorithm' check, and then
// again when the value is needed.
template<LegacyOverrideBuiltIns overrideBuiltins, class JSClass, class ItemAccessor>
static auto accessVisibleNamedProperty(JSC::JSGlobalObject& lexicalGlobalObject, JSClass& thisObject, JSC::PropertyName propertyName, ItemAccessor&& itemAccessor) -> decltype(itemAccessor(thisObject, propertyName))
{
    // NOTE: While it is not specified, a Symbol can never be a 'supported property
    // name' so we check that first.
    if (propertyName.isSymbol())
        return std::nullopt;

    // 1. If P is not a supported property name of O, then return false.
    auto result = itemAccessor(thisObject, propertyName);
    if (!result)
        return std::nullopt;

    // 2. If O has an own property named P, then return false.
    JSC::PropertySlot slot { &thisObject, JSC::PropertySlot::InternalMethodType::VMInquiry, &lexicalGlobalObject.vm() };
    if (JSC::JSObject::getOwnPropertySlot(&thisObject, &lexicalGlobalObject, propertyName, slot))
        return std::nullopt;

    // 3. If O implements an interface that has the [LegacyOverrideBuiltIns] extended attribute, then return true.
    if (overrideBuiltins == LegacyOverrideBuiltIns::Yes && !worldForDOMObject(thisObject).shouldDisableLegacyOverrideBuiltInsBehavior())
        return result;

    // 4. Initialize prototype to be the value of the internal [[Prototype]] property of O.
    // 5. While prototype is not null:
    //    1. If prototype is not a named properties object, and prototype has an own property named P, then return false.
    // FIXME: Implement checking for 'named properties object'.
    //    2. Set prototype to be the value of the internal [[Prototype]] property of prototype.
    auto prototype = thisObject.getPrototypeDirect();
    if (prototype.isObject() && JSC::asObject(prototype)->getPropertySlot(&lexicalGlobalObject, propertyName, slot))
        return std::nullopt;

    // 6. Return true.
    return result;
}

// This implements steps 2.2 through 2.5 of https://webidl.spec.whatwg.org/#legacy-platform-object-delete.
template<typename Functor> bool performLegacyPlatformObjectDeleteOperation(JSC::JSGlobalObject& lexicalGlobalObject, Functor&& functor)
{
    using ReturnType = std::invoke_result_t<Functor>;

    if constexpr (IsExceptionOr<ReturnType>) {
        auto result = functor();
        if (result.hasException()) {
            auto throwScope = DECLARE_THROW_SCOPE(JSC::getVM(&lexicalGlobalObject));
            propagateException(lexicalGlobalObject, throwScope, result.releaseException());
            return true;
        }
        
        if constexpr (std::is_same_v<ReturnType, ExceptionOr<bool>>)
            return result.releaseReturnValue();
        return true;
    }

    if constexpr (std::is_same_v<ReturnType, bool>)
        return functor();
    
    functor();
    return true;
}

}
