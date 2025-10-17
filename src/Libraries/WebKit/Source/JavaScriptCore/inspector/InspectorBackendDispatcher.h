/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 17, 2024.
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

#include "InspectorFrontendRouter.h"
#include "InspectorProtocolTypes.h"
#include <functional>
#include <wtf/Function.h>
#include <wtf/RefCounted.h>
#include <wtf/RefPtr.h>
#include <wtf/text/WTFString.h>

namespace Inspector {

class BackendDispatcher;

typedef String ErrorString;

class SupplementalBackendDispatcher : public RefCounted<SupplementalBackendDispatcher> {
public:
    JS_EXPORT_PRIVATE SupplementalBackendDispatcher(BackendDispatcher&);
    JS_EXPORT_PRIVATE virtual ~SupplementalBackendDispatcher();
    virtual void dispatch(long requestId, const String& method, Ref<JSON::Object>&& message) = 0;
protected:
    Ref<BackendDispatcher> m_backendDispatcher;
};

class BackendDispatcher : public RefCounted<BackendDispatcher> {
public:
    JS_EXPORT_PRIVATE static Ref<BackendDispatcher> create(Ref<FrontendRouter>&&);

    class CallbackBase : public RefCounted<CallbackBase> {
    public:
        JS_EXPORT_PRIVATE CallbackBase(Ref<BackendDispatcher>&&, long requestId);
        virtual ~CallbackBase() { }

        JS_EXPORT_PRIVATE bool isActive() const;
        void disable() { m_alreadySent = true; }

        JS_EXPORT_PRIVATE void sendSuccess(Ref<JSON::Object>&&);
        JS_EXPORT_PRIVATE void sendFailure(const ErrorString&);

    private:
        Ref<BackendDispatcher> m_backendDispatcher;
        long m_requestId;
        bool m_alreadySent { false };
    };

    bool isActive() const;

    bool hasProtocolErrors() const { return m_protocolErrors.size() > 0; }

    enum CommonErrorCode {
        ParseError = 0,
        InvalidRequest,
        MethodNotFound,
        InvalidParams,
        InternalError,
        ServerError
    };

    JS_EXPORT_PRIVATE void registerDispatcherForDomain(const String& domain, SupplementalBackendDispatcher*);
    JS_EXPORT_PRIVATE void dispatch(const String& message);

    // Note that 'unused' is a workaround so the compiler can pick the right sendResponse based on arity.
    // When <http://webkit.org/b/179847> is fixed or this class is renamed for the JSON::Object case,
    // then this alternate method with a dummy parameter can be removed in favor of the one without it.
    void sendResponse(long requestId, RefPtr<JSON::Object>&& result);
    void sendResponse(long requestId, RefPtr<JSON::Object>&& result, bool unused);
    void sendResponse(long requestId, Ref<JSON::Object>&& result);
    JS_EXPORT_PRIVATE void sendResponse(long requestId, Ref<JSON::Object>&& result, bool unused);
    JS_EXPORT_PRIVATE void sendPendingErrors();

    JS_EXPORT_PRIVATE void reportProtocolError(CommonErrorCode, const String& errorMessage);
    JS_EXPORT_PRIVATE void reportProtocolError(std::optional<long> relatedRequestId, CommonErrorCode, const String& errorMessage);

    JS_EXPORT_PRIVATE std::optional<bool> getBoolean(JSON::Object*, const String& name, bool required);
    JS_EXPORT_PRIVATE std::optional<int> getInteger(JSON::Object*, const String& name, bool required);
    JS_EXPORT_PRIVATE std::optional<double> getDouble(JSON::Object*, const String& name, bool required);
    JS_EXPORT_PRIVATE String getString(JSON::Object*, const String& name, bool required);
    JS_EXPORT_PRIVATE RefPtr<JSON::Value> getValue(JSON::Object*, const String& name, bool required);
    JS_EXPORT_PRIVATE RefPtr<JSON::Object> getObject(JSON::Object*, const String& name, bool required);
    JS_EXPORT_PRIVATE RefPtr<JSON::Array> getArray(JSON::Object*, const String& name, bool required);

private:
    BackendDispatcher(Ref<FrontendRouter>&&);

    template<typename T>
    WTF_INTERNAL T getPropertyValue(JSON::Object*, const String& name, bool required, std::function<T(JSON::Value&)> converter, ASCIILiteral typeName);

    Ref<FrontendRouter> m_frontendRouter;
    UncheckedKeyHashMap<String, SupplementalBackendDispatcher*> m_dispatchers;

    // Protocol errors reported for the top-level request being processed.
    // If processing a request triggers async responses, then any related errors will
    // be attributed to the top-level request, but generate separate error messages.
    Vector<std::tuple<CommonErrorCode, String>> m_protocolErrors;

    // For synchronously handled requests, avoid plumbing requestId through every
    // call that could potentially fail with a protocol error.
    std::optional<long> m_currentRequestId { std::nullopt };
};

} // namespace Inspector
