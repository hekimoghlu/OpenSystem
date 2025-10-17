/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 12, 2025.
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
#include "JSDOMWrapper.h"

#include "DOMWrapperWorld.h"
#include "JSDOMWindow.h"
#include "LocalDOMWindow.h"
#include "SerializedScriptValue.h"
#include "WebCoreJSClientData.h"
#include <JavaScriptCore/Error.h>

namespace WebCore {

STATIC_ASSERT_IS_TRIVIALLY_DESTRUCTIBLE(JSDOMObject);

JSDOMObject::JSDOMObject(JSC::Structure* structure, JSC::JSGlobalObject& globalObject)
    : Base(globalObject.vm(), structure)
{
    ASSERT(scriptExecutionContext() || globalObject.classInfo() == JSDOMWindow::info());
}

JSC::JSValue cloneAcrossWorlds(JSC::JSGlobalObject& lexicalGlobalObject, const JSDOMObject& owner, JSC::JSValue value)
{
    if (isWorldCompatible(lexicalGlobalObject, value))
        return value;
    // FIXME: Is it best to handle errors by returning null rather than throwing an exception?
    auto serializedValue = SerializedScriptValue::create(lexicalGlobalObject, value, SerializationForStorage::No, SerializationErrorMode::NonThrowing, SerializationContext::CloneAcrossWorlds);
    if (!serializedValue)
        return JSC::jsNull();
    // FIXME: Why is owner->globalObject() better than lexicalGlobalObject.lexicalGlobalObject() here?
    // Unlike this, isWorldCompatible uses lexicalGlobalObject.lexicalGlobalObject(); should the two match?
    return serializedValue->deserialize(lexicalGlobalObject, owner.globalObject());
}

} // namespace WebCore
