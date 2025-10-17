/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 21, 2025.
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
#include "PaymentResponse.h"

#if ENABLE(PAYMENT_REQUEST)

#include "Document.h"
#include "JSDOMPromiseDeferred.h"
#include "NotImplemented.h"
#include "PaymentComplete.h"
#include "PaymentCompleteDetails.h"
#include "PaymentRequest.h"
#include <JavaScriptCore/JSONObject.h>
#include <JavaScriptCore/ThrowScope.h>
#include <wtf/RunLoop.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(PaymentResponse);

PaymentResponse::PaymentResponse(ScriptExecutionContext* context, PaymentRequest& request)
    : ActiveDOMObject { context }
    , m_request { request }
{
}

void PaymentResponse::finishConstruction()
{
    ASSERT(!hasPendingActivity());
    m_pendingActivity = makePendingActivity(*this);
    suspendIfNeeded();
}

PaymentResponse::~PaymentResponse()
{
    ASSERT(!hasPendingActivity() || isContextStopped());
    ASSERT(!hasRetryPromise() || isContextStopped());
}

void PaymentResponse::setDetailsFunction(DetailsFunction&& detailsFunction)
{
    m_detailsFunction = WTFMove(detailsFunction);
    m_cachedDetails.clear();
}

void PaymentResponse::complete(Document& document, std::optional<PaymentComplete>&& result, std::optional<PaymentCompleteDetails>&& details, DOMPromiseDeferred<void>&& promise)
{
    if (m_state == State::Stopped || !m_request) {
        promise.reject(Exception { ExceptionCode::AbortError });
        return;
    }

    if (m_state == State::Completed || m_retryPromise) {
        promise.reject(Exception { ExceptionCode::InvalidStateError });
        return;
    }

    String serializedData;
    if (details) {
        if (auto data = details->data) {
            auto throwScope = DECLARE_THROW_SCOPE(document.globalObject()->vm());

            serializedData = JSONStringify(document.globalObject(), data.get(), 0);
            if (throwScope.exception()) {
                promise.reject(Exception { ExceptionCode::ExistingExceptionError });
                return;
            }
        }
    }

    auto exception = m_request->complete(document, WTFMove(result), WTFMove(serializedData));
    if (!exception.hasException()) {
        ASSERT(hasPendingActivity());
        ASSERT(m_state == State::Created);
        m_pendingActivity = nullptr;
        m_state = State::Completed;
    }
    promise.settle(WTFMove(exception));
}

void PaymentResponse::retry(PaymentValidationErrors&& errors, DOMPromiseDeferred<void>&& promise)
{
    if (m_state == State::Stopped || !m_request) {
        promise.reject(Exception { ExceptionCode::AbortError });
        return;
    }

    if (m_state == State::Completed || m_retryPromise) {
        promise.reject(Exception { ExceptionCode::InvalidStateError });
        return;
    }

    ASSERT(hasPendingActivity());
    ASSERT(m_state == State::Created);

    auto exception = m_request->retry(WTFMove(errors));
    if (exception.hasException()) {
        promise.reject(exception.releaseException());
        return;
    }

    m_retryPromise = makeUnique<DOMPromiseDeferred<void>>(WTFMove(promise));
}

void PaymentResponse::abortWithException(Exception&& exception)
{
    settleRetryPromise(WTFMove(exception));
    m_pendingActivity = nullptr;
    m_state = State::Completed;
}

void PaymentResponse::settleRetryPromise(ExceptionOr<void>&& result)
{
    if (!m_retryPromise)
        return;

    ASSERT(hasPendingActivity());
    ASSERT(m_state == State::Created || m_state == State::Stopped);
    m_retryPromise->settle(WTFMove(result));
    m_retryPromise = nullptr;
}

void PaymentResponse::stop()
{
    queueTaskKeepingObjectAlive(*this, TaskSource::Payment, [this, pendingActivity = std::exchange(m_pendingActivity, nullptr)] {
        settleRetryPromise(Exception { ExceptionCode::AbortError });
    });
    m_state = State::Stopped;
}

void PaymentResponse::suspend(ReasonForSuspension reason)
{
    if (reason != ReasonForSuspension::BackForwardCache)
        return;

    if (m_state != State::Created) {
        ASSERT(!hasPendingActivity());
        ASSERT(!m_retryPromise);
        return;
    }

    stop();
}

} // namespace WebCore

#endif // ENABLE(PAYMENT_REQUEST)
