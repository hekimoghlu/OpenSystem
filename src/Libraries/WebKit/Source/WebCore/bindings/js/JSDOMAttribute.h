/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 6, 2025.
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

#include "JSDOMCastThisValue.h"
#include "JSDOMExceptionHandling.h"
#include <JavaScriptCore/Error.h>

namespace WebCore {

template<typename JSClass>
class IDLAttribute {
public:
    using Setter = bool(JSC::JSGlobalObject&, JSClass&, JSC::JSValue);
    using SetterPassingPropertyName = bool(JSC::JSGlobalObject&, JSClass&, JSC::JSValue, JSC::PropertyName);
    using StaticSetter = bool(JSC::JSGlobalObject&, JSC::JSValue);
    using Getter = JSC::JSValue(JSC::JSGlobalObject&, JSClass&);
    using GetterPassingPropertyName = JSC::JSValue(JSC::JSGlobalObject&, JSClass&, JSC::PropertyName);
    using StaticGetter = JSC::JSValue(JSC::JSGlobalObject&);

    template<Setter setter, CastedThisErrorBehavior shouldThrow = CastedThisErrorBehavior::Throw>
    static bool set(JSC::JSGlobalObject& lexicalGlobalObject, JSC::EncodedJSValue thisValue, JSC::EncodedJSValue encodedValue, JSC::PropertyName attributeName)
    {
        auto throwScope = DECLARE_THROW_SCOPE(JSC::getVM(&lexicalGlobalObject));

        auto* thisObject = castThisValue<JSClass>(lexicalGlobalObject, JSC::JSValue::decode(thisValue));
        if (UNLIKELY(!thisObject)) {
            if constexpr (shouldThrow == CastedThisErrorBehavior::Throw)
                return JSC::throwVMDOMAttributeSetterTypeError(&lexicalGlobalObject, throwScope, JSClass::info(), attributeName);
            else
                return false;
        }

        RELEASE_AND_RETURN(throwScope, (setter(lexicalGlobalObject, *thisObject, JSC::JSValue::decode(encodedValue))));
    }

    // FIXME: FIXME: This can be merged with `set` if we replace the explicit setter function template parameter with a generic lambda.
    template<SetterPassingPropertyName setter, CastedThisErrorBehavior shouldThrow = CastedThisErrorBehavior::Throw>
    static bool setPassingPropertyName(JSC::JSGlobalObject& lexicalGlobalObject, JSC::EncodedJSValue thisValue, JSC::EncodedJSValue encodedValue, JSC::PropertyName attributeName)
    {
        auto throwScope = DECLARE_THROW_SCOPE(JSC::getVM(&lexicalGlobalObject));

        auto* thisObject = castThisValue<JSClass>(lexicalGlobalObject, JSC::JSValue::decode(thisValue));
        if (UNLIKELY(!thisObject)) {
            if constexpr (shouldThrow == CastedThisErrorBehavior::Throw)
                return JSC::throwVMDOMAttributeSetterTypeError(&lexicalGlobalObject, throwScope, JSClass::info(), attributeName);
            else
                return false;
        }

        RELEASE_AND_RETURN(throwScope, (setter(lexicalGlobalObject, *thisObject, JSC::JSValue::decode(encodedValue), attributeName)));
    }

    template<StaticSetter setter, CastedThisErrorBehavior shouldThrow = CastedThisErrorBehavior::Throw>
    static bool setStatic(JSC::JSGlobalObject& lexicalGlobalObject, JSC::EncodedJSValue, JSC::EncodedJSValue encodedValue, JSC::PropertyName)
    {
        return setter(lexicalGlobalObject, JSC::JSValue::decode(encodedValue));
    }
    
    template<Getter getter, CastedThisErrorBehavior shouldThrow = CastedThisErrorBehavior::Throw>
    static JSC::EncodedJSValue get(JSC::JSGlobalObject& lexicalGlobalObject, JSC::EncodedJSValue thisValue, JSC::PropertyName attributeName)
    {
        auto throwScope = DECLARE_THROW_SCOPE(JSC::getVM(&lexicalGlobalObject));

        if constexpr (shouldThrow == CastedThisErrorBehavior::Assert) {
            ASSERT(castThisValue<JSClass>(lexicalGlobalObject, JSC::JSValue::decode(thisValue)));
            auto* thisObject = JSC::jsCast<JSClass*>(JSC::JSValue::decode(thisValue));
            RELEASE_AND_RETURN(throwScope, (JSC::JSValue::encode(getter(lexicalGlobalObject, *thisObject))));
        } else {
            auto* thisObject = castThisValue<JSClass>(lexicalGlobalObject, JSC::JSValue::decode(thisValue));
            if (UNLIKELY(!thisObject)) {
                if constexpr (shouldThrow == CastedThisErrorBehavior::Throw)
                    return JSC::throwVMDOMAttributeGetterTypeError(&lexicalGlobalObject, throwScope, JSClass::info(), attributeName);
                else if constexpr (shouldThrow == CastedThisErrorBehavior::RejectPromise)
                    RELEASE_AND_RETURN(throwScope, rejectPromiseWithGetterTypeError(lexicalGlobalObject, JSClass::info(), attributeName));
                else
                    return JSC::JSValue::encode(JSC::jsUndefined());
            }

            RELEASE_AND_RETURN(throwScope, (JSC::JSValue::encode(getter(lexicalGlobalObject, *thisObject))));
        }
    }

    // FIXME: This can be merged with `get` if we replace the explicit setter function template parameter with a generic lambda.
    template<GetterPassingPropertyName getter, CastedThisErrorBehavior shouldThrow = CastedThisErrorBehavior::Throw>
    static JSC::EncodedJSValue getPassingPropertyName(JSC::JSGlobalObject& lexicalGlobalObject, JSC::EncodedJSValue thisValue, JSC::PropertyName attributeName)
    {
        auto throwScope = DECLARE_THROW_SCOPE(JSC::getVM(&lexicalGlobalObject));

        if constexpr (shouldThrow == CastedThisErrorBehavior::Assert) {
            ASSERT(castThisValue<JSClass>(lexicalGlobalObject, JSC::JSValue::decode(thisValue)));
            auto* thisObject = JSC::jsCast<JSClass*>(JSC::JSValue::decode(thisValue));
            RELEASE_AND_RETURN(throwScope, (JSC::JSValue::encode(getter(lexicalGlobalObject, *thisObject, attributeName))));
        } else {
            auto* thisObject = castThisValue<JSClass>(lexicalGlobalObject, JSC::JSValue::decode(thisValue));
            if (UNLIKELY(!thisObject)) {
                if constexpr (shouldThrow == CastedThisErrorBehavior::Throw)
                    return JSC::throwVMDOMAttributeGetterTypeError(&lexicalGlobalObject, throwScope, JSClass::info(), attributeName);
                else if constexpr (shouldThrow == CastedThisErrorBehavior::RejectPromise)
                    RELEASE_AND_RETURN(throwScope, rejectPromiseWithGetterTypeError(lexicalGlobalObject, JSClass::info(), attributeName));
                else
                    return JSC::JSValue::encode(JSC::jsUndefined());
            }

            RELEASE_AND_RETURN(throwScope, (JSC::JSValue::encode(getter(lexicalGlobalObject, *thisObject, attributeName))));
        }
    }

    template<StaticGetter getter, CastedThisErrorBehavior shouldThrow = CastedThisErrorBehavior::Throw>
    static JSC::EncodedJSValue getStatic(JSC::JSGlobalObject& lexicalGlobalObject, JSC::EncodedJSValue, JSC::PropertyName)
    {
        return JSC::JSValue::encode(getter(lexicalGlobalObject));
    }
};

} // namespace WebCore
