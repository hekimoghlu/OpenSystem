/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 22, 2022.
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

namespace WebCore {

template<typename JSClass>
class IDLOperation {
public:
    using ClassParameter = JSClass*;
    using Operation = JSC::EncodedJSValue(JSC::JSGlobalObject*, JSC::CallFrame*, ClassParameter);
    using StaticOperation = JSC::EncodedJSValue(JSC::JSGlobalObject*, JSC::CallFrame*);

    // FIXME: Remove this after FunctionCallResolveNode is fixed not to pass resolved scope as |this| value.
    // https://bugs.webkit.org/show_bug.cgi?id=225397
    static JSClass* cast(JSC::JSGlobalObject& lexicalGlobalObject, JSC::CallFrame& callFrame)
    {
        if constexpr (std::is_base_of_v<JSDOMGlobalObject, JSClass>)
            return castThisValue<JSClass>(lexicalGlobalObject, callFrame.thisValue().toThis(&lexicalGlobalObject, JSC::ECMAMode::strict()));
        else
            return castThisValue<JSClass>(lexicalGlobalObject, callFrame.thisValue());
    }

    template<Operation operation, CastedThisErrorBehavior shouldThrow = CastedThisErrorBehavior::Throw>
    static JSC::EncodedJSValue call(JSC::JSGlobalObject& lexicalGlobalObject, JSC::CallFrame& callFrame, const char* operationName)
    {
        auto throwScope = DECLARE_THROW_SCOPE(JSC::getVM(&lexicalGlobalObject));
        
        auto* thisObject = cast(lexicalGlobalObject, callFrame);
        if constexpr (shouldThrow != CastedThisErrorBehavior::Assert) {
            if (UNLIKELY(!thisObject))
                return throwThisTypeError(lexicalGlobalObject, throwScope, JSClass::info()->className, operationName);
        } else
            ASSERT(thisObject);

        ASSERT_GC_OBJECT_INHERITS(thisObject, JSClass::info());
        
        // FIXME: We should refactor the binding generated code to use references for lexicalGlobalObject and thisObject.
        RELEASE_AND_RETURN(throwScope, (operation(&lexicalGlobalObject, &callFrame, thisObject)));
    }

    template<StaticOperation operation, CastedThisErrorBehavior shouldThrow = CastedThisErrorBehavior::Throw>
    static JSC::EncodedJSValue callStatic(JSC::JSGlobalObject& lexicalGlobalObject, JSC::CallFrame& callFrame, const char*)
    {
        // FIXME: We should refactor the binding generated code to use references for lexicalGlobalObject.
        return operation(&lexicalGlobalObject, &callFrame);
    }
};

} // namespace WebCore
