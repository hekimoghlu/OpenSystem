/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 30, 2024.
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
#include "CredentialRequestCoordinator.h"

#if ENABLE(WEB_AUTHN)

#include "AbortSignal.h"
#include "CredentialRequestCoordinatorClient.h"
#include "CredentialRequestOptions.h"
#include "Document.h"
#include "DocumentInlines.h"
#include "FrameDestructionObserverInlines.h"
#include "JSDOMPromiseDeferred.h"
#include "VisibilityState.h"
#include <wtf/Logger.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(CredentialRequestCoordinator);
WTF_MAKE_TZONE_ALLOCATED_IMPL(CredentialRequestCoordinatorClient);

CredentialRequestCoordinator::CredentialRequestCoordinator(std::unique_ptr<CredentialRequestCoordinatorClient>&& client)
    : m_client(WTFMove(client))
{
}

void CredentialRequestCoordinator::discoverFromExternalSource(const Document& document, CredentialRequestOptions&& requestOptions, CredentialPromise&& promise)
{
    RefPtr window = document.protectedWindow();
    if (!m_client || !window) {
        LOG_ERROR("No client or window found");
        promise.reject(Exception { ExceptionCode::UnknownError, "Unknown internal error."_s });
        return;
    }

    if (!document.hasFocus()) {
        promise.reject(Exception { ExceptionCode::NotAllowedError, "The document is not focused."_s });
        return;
    }

    if (document.visibilityState() != VisibilityState::Visible) {
        promise.reject(Exception { ExceptionCode::NotAllowedError, "The document is not visible."_s });
        return;
    }

    const auto& options = requestOptions.digital.value();

    if (!options.requests.size()) {
        promise.reject(Exception { ExceptionCode::TypeError, "Must make at least one request."_s });
        return;
    }

    if (!window->consumeTransientActivation()) {
        promise.reject(Exception { ExceptionCode::NotAllowedError, "Calling get() needs to be triggered by an activation triggering user event."_s });
        return;
    }

    if (requestOptions.signal) {
        requestOptions.signal->addAlgorithm([this](JSC::JSValue) mutable {
            if (!this->m_client)
                return;

            ASSERT(!this->m_isCancelling);

            this->m_isCancelling = true;
            this->m_client->cancel([this]() mutable {
                this->m_isCancelling = false;
                if (auto queuedRequest = WTFMove(this->m_queuedRequest))
                    queuedRequest();
            });
        });
    }

    auto callback = [promise = WTFMove(promise), abortSignal = WTFMove(requestOptions.signal)](ExceptionData&& exception) mutable {
        if (abortSignal && abortSignal->aborted()) {
            LOG_ERROR("Request aborted by AbortSignal");
            promise.reject(Exception { ExceptionCode::AbortError, "Aborted by AbortSignal."_s });
            return;
        }
        ASSERT(!exception.message.isNull());
        LOG_ERROR("Exception occurred: %s", exception.message.utf8().data());
        promise.reject(exception.toException());
    };

    RefPtr frame = document.frame();
    ASSERT(frame);

    if (m_isCancelling) {
        m_queuedRequest = [this, weakFrame = WeakPtr(*frame), requestOptions = WTFMove(requestOptions), callback = WTFMove(callback)]() mutable {
            if (!this->m_client || !weakFrame) {
                LOG_ERROR("No this, or no frame, or no client found");
                return;
            }
            const auto options = requestOptions.digital.value();
            this->m_client->requestDigitalCredential(*weakFrame, options, WTFMove(callback));
        };
        return;
    }

    m_client->requestDigitalCredential(*frame, options, WTFMove(callback));
}

} // namespace WebCore

#endif // ENABLE(WEB_AUTHN)
