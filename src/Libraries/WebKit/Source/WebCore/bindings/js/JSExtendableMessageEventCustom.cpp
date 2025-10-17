/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 14, 2023.
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

#include "JSExtendableMessageEvent.h"

#include "JSDOMConstructor.h"
#include "JSDOMConvertInterface.h"
#include "JSDOMConvertSequences.h"
#include "JSDOMConvertStrings.h"
#include "JSMessagePort.h"

namespace WebCore {

using namespace JSC;

JSC::EncodedJSValue constructJSExtendableMessageEvent(JSC::JSGlobalObject* lexicalGlobalObject, JSC::CallFrame& callFrame)
{
    VM& vm = lexicalGlobalObject->vm();
    auto throwScope = DECLARE_THROW_SCOPE(vm);

    if (UNLIKELY(callFrame.argumentCount() < 1))
        return throwVMError(lexicalGlobalObject, throwScope, createNotEnoughArgumentsError(lexicalGlobalObject));

    auto type = convert<IDLAtomStringAdaptor<IDLDOMString>>(*lexicalGlobalObject, callFrame.uncheckedArgument(0));
    if (UNLIKELY(type.hasException(throwScope)))
        return encodedJSValue();

    auto eventInitDict = convert<IDLDictionary<ExtendableMessageEvent::Init>>(*lexicalGlobalObject, callFrame.argument(1));
    if (UNLIKELY(eventInitDict.hasException(throwScope)))
        return encodedJSValue();

    auto object = ExtendableMessageEvent::create(*lexicalGlobalObject, type.releaseReturnValue(), eventInitDict.releaseReturnValue());
    RETURN_IF_EXCEPTION(throwScope, encodedJSValue());

    return JSValue::encode(object.strongWrapper.get());
}

JSC::JSValue JSExtendableMessageEvent::ports(JSC::JSGlobalObject& lexicalGlobalObject) const
{
    auto throwScope = DECLARE_THROW_SCOPE(lexicalGlobalObject.vm());
    return cachedPropertyValue(throwScope, lexicalGlobalObject, *this, wrapped().cachedPorts(), [&](JSC::ThrowScope& throwScope) {
        return toJS<IDLFrozenArray<IDLInterface<MessagePort>>>(lexicalGlobalObject, *globalObject(), throwScope, wrapped().ports());
    });
}

template<typename Visitor>
void JSExtendableMessageEvent::visitAdditionalChildren(Visitor& visitor)
{
    wrapped().data().visit(visitor);
    wrapped().cachedPorts().visit(visitor);
}

DEFINE_VISIT_ADDITIONAL_CHILDREN(JSExtendableMessageEvent);

}
