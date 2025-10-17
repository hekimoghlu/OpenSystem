/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 24, 2021.
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
#include "JSNavigator.h"

#include "WebCoreJSBuiltinInternals.h"
#include "WebCoreJSClientData.h"
#include "WebCoreOpaqueRootInlines.h"
#include <JavaScriptCore/CatchScope.h>
#include <JavaScriptCore/JSCJSValue.h>

namespace WebCore {

template<typename Visitor>
void JSNavigator::visitAdditionalChildren(Visitor& visitor)
{
    addWebCoreOpaqueRoot(visitor, static_cast<NavigatorBase&>(wrapped()));
}

DEFINE_VISIT_ADDITIONAL_CHILDREN(JSNavigator);

#if ENABLE(MEDIA_STREAM)
JSC::JSValue JSNavigator::getUserMedia(JSC::JSGlobalObject& lexicalGlobalObject, JSC::CallFrame& callFrame)
{
    auto* function = globalObject()->builtinInternalFunctions().jsDOMBindingInternals().m_getUserMediaShimFunction.get();
    ASSERT(function);

    auto callData = JSC::getCallData(function);
    ASSERT(callData.type != JSC::CallData::Type::None);
    JSC::MarkedArgumentBuffer arguments;
    arguments.ensureCapacity(callFrame.argumentCount());
    for (size_t cptr = 0; cptr < callFrame.argumentCount(); ++cptr)
        arguments.append(callFrame.uncheckedArgument(cptr));
    ASSERT(!arguments.hasOverflowed());
    JSC::call(&lexicalGlobalObject, function, callData, this, arguments);
    return JSC::jsUndefined();
}
#endif

}
