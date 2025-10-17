/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 5, 2021.
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
#pragma once

#include "ExceptionDetails.h"
#include <wtf/CompletionHandler.h>
#include <wtf/Expected.h>
#include <wtf/JSONValues.h>
#include <wtf/RefCounted.h>
#include <wtf/WeakPtr.h>
#include <wtf/text/WTFString.h>

namespace JSC {
class JSValue;
}

namespace WebCore {

class DOMPromise;
class JSDOMGlobalObject;
class Page;

class InspectorFrontendAPIDispatcher final
    : public RefCounted<InspectorFrontendAPIDispatcher>
    , public CanMakeWeakPtr<InspectorFrontendAPIDispatcher> {
public:
    enum class EvaluationError { ExecutionSuspended, ContextDestroyed, InternalError };
    using ValueOrException = Expected<JSC::JSValue, ExceptionDetails>;
    using EvaluationResult = Expected<ValueOrException, EvaluationError>;
    using EvaluationResultHandler = CompletionHandler<void(EvaluationResult)>;

    enum class UnsuspendSoon : bool { No, Yes };
    
    WEBCORE_EXPORT ~InspectorFrontendAPIDispatcher();

    static Ref<InspectorFrontendAPIDispatcher> create(Page& page)
    {
        return adoptRef(*new InspectorFrontendAPIDispatcher(page));
    }

    WEBCORE_EXPORT void reset();
    WEBCORE_EXPORT void frontendLoaded();

    // If it's not currently safe to evaluate JavaScript on the frontend page, the
    // dispatcher will become suspended and dispatch any queued evaluations when unsuspended.
    WEBCORE_EXPORT void suspend(UnsuspendSoon = UnsuspendSoon::No);
    WEBCORE_EXPORT void unsuspend();
    bool isSuspended() const { return m_suspended; }

    EvaluationResult dispatchCommandWithResultSync(const String& command, Vector<Ref<JSON::Value>>&& arguments = { });
    WEBCORE_EXPORT void dispatchCommandWithResultAsync(const String& command, Vector<Ref<JSON::Value>>&& arguments = { }, EvaluationResultHandler&& handler = { });

    // Used to forward messages from the backend connection to the frontend.
    WEBCORE_EXPORT void dispatchMessageAsync(const String& message);

    WEBCORE_EXPORT void evaluateExpressionForTesting(const String&);
    
    // Convenience method to obtain a JSDOMGlobalObject for the frontend page.
    // This is used to convert between C++ values and frontend JSC::JSValue objects.
    WEBCORE_EXPORT JSDOMGlobalObject* frontendGlobalObject();
private:
    WEBCORE_EXPORT InspectorFrontendAPIDispatcher(Page&);

    void evaluateOrQueueExpression(const String&, EvaluationResultHandler&& handler = { });
    void evaluateQueuedExpressions();
    void invalidateQueuedExpressions();
    void invalidatePendingResponses();
    ValueOrException evaluateExpression(const String&);

    WeakPtr<Page> m_frontendPage;
    Vector<std::pair<String, EvaluationResultHandler>> m_queuedEvaluations;
    HashMap<Ref<DOMPromise>, EvaluationResultHandler> m_pendingResponses;
    bool m_frontendLoaded { false };
    bool m_suspended { false };
};

} // namespace WebCore
