/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 13, 2021.
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
#include "UserMessageHandler.h"

#if ENABLE(USER_MESSAGE_HANDLERS)

#include "JSDOMPromiseDeferred.h"
#include "LocalFrame.h"
#include "SerializedScriptValue.h"
#include <JavaScriptCore/JSCJSValue.h>

namespace WebCore {

UserMessageHandler::UserMessageHandler(LocalFrame& frame, UserMessageHandlerDescriptor& descriptor)
    : FrameDestructionObserver(&frame)
    , m_descriptor(&descriptor)
{
}

UserMessageHandler::~UserMessageHandler() = default;

ExceptionOr<void> UserMessageHandler::postMessage(RefPtr<SerializedScriptValue>&& value, Ref<DeferredPromise>&& promise)
{
    // Check to see if the descriptor has been removed. This can happen if the host application has
    // removed the named message handler at the WebKit2 API level.
    if (!m_descriptor) {
        promise->reject(Exception { ExceptionCode::InvalidAccessError });
        return Exception { ExceptionCode::InvalidAccessError };
    }

    m_descriptor->didPostMessage(*this, value.get(), [promise = WTFMove(promise)](SerializedScriptValue* result, const String& errorMessage) {
        auto* globalObject = promise->globalObject();
        if (!globalObject)
            return;

        if (!errorMessage.isNull()) {
            JSC::JSLockHolder lock(globalObject);
            promise->reject<IDLAny>(JSC::createError(globalObject, errorMessage));
            return;
        }

        ASSERT(result);
        promise->resolve<IDLAny>(result->deserialize(*globalObject, globalObject));
    });
    return { };
}

} // namespace WebCore

#endif // ENABLE(USER_MESSAGE_HANDLERS)
