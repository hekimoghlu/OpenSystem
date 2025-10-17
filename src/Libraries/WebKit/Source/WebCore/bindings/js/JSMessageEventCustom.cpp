/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 10, 2025.
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
#include "JSMessageEvent.h"

#include "JSBlob.h"
#include "JSDOMBinding.h"
#include "JSDOMConvert.h"
#include "JSEventTarget.h"
#include "JSMessagePort.h"
#include <JavaScriptCore/JSArray.h>
#include <JavaScriptCore/JSArrayBuffer.h>

namespace WebCore {

JSC::JSValue JSMessageEvent::ports(JSC::JSGlobalObject& lexicalGlobalObject) const
{
    auto throwScope = DECLARE_THROW_SCOPE(lexicalGlobalObject.vm());
    return cachedPropertyValue(throwScope, lexicalGlobalObject, *this, wrapped().cachedPorts(), [&](JSC::ThrowScope& throwScope) {
        return toJS<IDLFrozenArray<IDLInterface<MessagePort>>>(lexicalGlobalObject, *globalObject(), throwScope, wrapped().ports());
    });
}

JSC::JSValue JSMessageEvent::data(JSC::JSGlobalObject& lexicalGlobalObject) const
{
    auto throwScope = DECLARE_THROW_SCOPE(lexicalGlobalObject.vm());
    return cachedPropertyValue(throwScope, lexicalGlobalObject, *this, wrapped().cachedData(), [this, &lexicalGlobalObject](JSC::ThrowScope&) {
        return WTF::switchOn(wrapped().data(), [this] (MessageEvent::JSValueTag) -> JSC::JSValue {
            return wrapped().jsData().getValue(JSC::jsNull());
        }, [this, &lexicalGlobalObject] (const Ref<SerializedScriptValue>& data) {
            // FIXME: Is it best to handle errors by returning null rather than throwing an exception?
            return data->deserialize(lexicalGlobalObject, globalObject(), wrapped().ports(), SerializationErrorMode::NonThrowing);
        }, [&lexicalGlobalObject] (const String& data) {
            return toJS<IDLDOMString>(lexicalGlobalObject, data);
        }, [this, &lexicalGlobalObject] (const Ref<Blob>& data) {
            return toJS<IDLInterface<Blob>>(lexicalGlobalObject, *globalObject(), data);
        }, [this, &lexicalGlobalObject] (const Ref<ArrayBuffer>& data) {
            return toJS<IDLInterface<ArrayBuffer>>(lexicalGlobalObject, *globalObject(), data);
        });
    });
}

template<typename Visitor>
void JSMessageEvent::visitAdditionalChildren(Visitor& visitor)
{
    wrapped().jsData().visit(visitor);
    wrapped().cachedData().visit(visitor);
    wrapped().cachedPorts().visit(visitor);
}

DEFINE_VISIT_ADDITIONAL_CHILDREN(JSMessageEvent);

} // namespace WebCore
