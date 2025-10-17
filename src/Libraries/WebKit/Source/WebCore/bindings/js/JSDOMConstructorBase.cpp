/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 2, 2022.
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
#include "JSDOMConstructor.h"

#include "WebCoreJSClientData.h"
#include <JavaScriptCore/JSCInlines.h>

namespace WebCore {
using namespace JSC;

STATIC_ASSERT_IS_TRIVIALLY_DESTRUCTIBLE(JSDOMConstructorBase);

JSC_DEFINE_HOST_FUNCTION(callThrowTypeErrorForJSDOMConstructor, (JSGlobalObject* globalObject, CallFrame*))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);
    throwTypeError(globalObject, scope, "Constructor requires 'new' operator"_s);
    return JSValue::encode(jsNull());
}

JSC_DEFINE_HOST_FUNCTION(callThrowTypeErrorForJSDOMConstructorNotConstructable, (JSC::JSGlobalObject* globalObject, JSC::CallFrame*))
{
    JSC::VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);
    JSC::throwTypeError(globalObject, scope, "Illegal constructor"_s);
    return JSC::JSValue::encode(JSC::jsNull());
}

JSC::GCClient::IsoSubspace* JSDOMConstructorBase::subspaceForImpl(JSC::VM& vm)
{
    return &static_cast<JSVMClientData*>(vm.clientData)->domConstructorSpace();
}

} // namespace WebCore
