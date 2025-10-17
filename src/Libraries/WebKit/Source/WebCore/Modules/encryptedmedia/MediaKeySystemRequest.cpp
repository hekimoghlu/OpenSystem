/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 3, 2024.
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
#include "MediaKeySystemRequest.h"

#if ENABLE(ENCRYPTED_MEDIA)

#include "Document.h"
#include "JSDOMPromiseDeferred.h"
#include "JSMediaKeySystemAccess.h"
#include "LocalFrame.h"
#include "Logging.h"
#include "MediaKeySystemController.h"
#include "Page.h"
#include "PlatformMediaSessionManager.h"
#include "Settings.h"
#include "WindowEventLoop.h"

namespace WebCore {

Ref<MediaKeySystemRequest> MediaKeySystemRequest::create(Document& document, const String& keySystem, Ref<DeferredPromise>&& promise)
{
    auto result = adoptRef(*new MediaKeySystemRequest(document, keySystem, WTFMove(promise)));
    result->suspendIfNeeded();
    return result;
}

MediaKeySystemRequest::MediaKeySystemRequest(Document& document, const String& keySystem, Ref<DeferredPromise>&& promise)
    : ActiveDOMObject(document)
    , m_keySystem(keySystem)
    , m_promise(WTFMove(promise))
{
}

MediaKeySystemRequest::~MediaKeySystemRequest()
{
    if (m_allowCompletionHandler)
        m_allowCompletionHandler(WTFMove(m_promise));
}

SecurityOrigin* MediaKeySystemRequest::topLevelDocumentOrigin() const
{
    RefPtr context = scriptExecutionContext();
    return context ? &context->topOrigin() : nullptr;
}

void MediaKeySystemRequest::start()
{
    RefPtr context = scriptExecutionContext();
    ASSERT(context);
    if (!context) {
        deny();
        return;
    }

    auto& document = downcast<Document>(*context);
    auto* controller = MediaKeySystemController::from(document.page());
    if (!controller) {
        deny();
        return;
    }

    controller->requestMediaKeySystem(*this);
}

void MediaKeySystemRequest::allow()
{
    if (!scriptExecutionContext())
        return;

    queueTaskKeepingObjectAlive(*this, TaskSource::UserInteraction, [this] {
        if (auto allowCompletionHandler = std::exchange(m_allowCompletionHandler, { }))
            allowCompletionHandler(WTFMove(m_promise));
    });
}

void MediaKeySystemRequest::deny(const String& message)
{
    if (!scriptExecutionContext())
        return;

    ExceptionCode code = ExceptionCode::NotSupportedError;
    if (!message.isEmpty())
        m_promise->reject(code, message);
    else
        m_promise->reject(code);
}

void MediaKeySystemRequest::stop()
{
    auto& document = downcast<Document>(*scriptExecutionContext());
    if (auto* controller = MediaKeySystemController::from(document.page()))
        controller->cancelMediaKeySystemRequest(*this);
}

Document* MediaKeySystemRequest::document() const
{
    return downcast<Document>(scriptExecutionContext());
}

} // namespace WebCore

#endif // ENABLE(ENCRYPTED_MEDIA)
