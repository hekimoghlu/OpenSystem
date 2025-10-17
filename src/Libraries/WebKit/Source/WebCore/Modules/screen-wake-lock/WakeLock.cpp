/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 19, 2023.
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
#include "WakeLock.h"

#include "DocumentInlines.h"
#include "EventLoop.h"
#include "Exception.h"
#include "JSDOMPromiseDeferred.h"
#include "JSWakeLockSentinel.h"
#include "LocalDOMWindow.h"
#include "Page.h"
#include "PermissionController.h"
#include "PermissionQuerySource.h"
#include "PermissionState.h"
#include "PermissionsPolicy.h"
#include "VisibilityState.h"
#include "WakeLockManager.h"
#include "WakeLockSentinel.h"

namespace WebCore {

WakeLock::WakeLock(Document* document)
    : ContextDestructionObserver(document)
{
}

// https://www.w3.org/TR/screen-wake-lock/#the-request-method
void WakeLock::request(WakeLockType lockType, Ref<DeferredPromise>&& promise)
{
    RefPtr document = this->document();
    if (!document || !document->isFullyActive() || !document->page()) {
        promise->reject(Exception { ExceptionCode::NotAllowedError, "Document is not fully active"_s });
        return;
    }
    if (!PermissionsPolicy::isFeatureEnabled(PermissionsPolicy::Feature::ScreenWakeLock, *document, PermissionsPolicy::ShouldReportViolation::Yes)) {
        promise->reject(Exception { ExceptionCode::NotAllowedError, "'screen-wake-lock' is not allowed by Feature-Policy"_s });
        return;
    }
    if (document->visibilityState() == VisibilityState::Hidden) {
        promise->reject(Exception { ExceptionCode::NotAllowedError, "Document is hidden"_s });
        return;
    }

    // FIXME: The permission check can likely be dropped once the specification gets updated to only
    // require transient activation (https://github.com/w3c/screen-wake-lock/pull/326).
    bool hasTransientActivation = document->domWindow() && document->domWindow()->hasTransientActivation();
    PermissionController::shared().query(document->clientOrigin(), PermissionDescriptor { PermissionName::ScreenWakeLock }, *document->page(), PermissionQuerySource::Window, [this, protectedThis = Ref { *this }, document = Ref { *document }, hasTransientActivation, promise = WTFMove(promise), lockType](std::optional<PermissionState> permission) mutable {
        if (!permission || *permission == PermissionState::Prompt) {
            if (hasTransientActivation || m_wasPreviouslyAuthorizedDueToTransientActivation) {
                m_wasPreviouslyAuthorizedDueToTransientActivation = true;
                permission = PermissionState::Granted;
            } else
                permission = PermissionState::Denied;
        } else if (*permission == PermissionState::Denied)
            m_wasPreviouslyAuthorizedDueToTransientActivation = false;
        document->eventLoop().queueTask(TaskSource::ScreenWakelock, [protectedThis = WTFMove(protectedThis), document = WTFMove(document), promise = WTFMove(promise), lockType, permission]() mutable {
            if (permission == PermissionState::Denied) {
                promise->reject(Exception { ExceptionCode::NotAllowedError, "Permission was denied"_s });
                return;
            }
            if (!document->isFullyActive()) {
                promise->reject(Exception { ExceptionCode::NotAllowedError, "Document is not fully active"_s });
                return;
            }
            if (document->visibilityState() == VisibilityState::Hidden) {
                promise->reject(Exception { ExceptionCode::NotAllowedError, "Document is hidden"_s });
                return;
            }
            auto lock = WakeLockSentinel::create(document, lockType);
            promise->resolve<IDLInterface<WakeLockSentinel>>(lock.get());
            document->wakeLockManager().addWakeLock(WTFMove(lock), document->pageID());
        });
    });
}

Document* WakeLock::document()
{
    return downcast<Document>(scriptExecutionContext());
}

} // namespace WebCore
