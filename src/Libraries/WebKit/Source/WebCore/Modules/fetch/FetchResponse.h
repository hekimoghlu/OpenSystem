/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 9, 2023.
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

#include "FetchBodyOwner.h"
#include "FetchHeaders.h"
#include "HTTPStatusCodes.h"
#include "ReadableStreamSink.h"
#include "ResourceResponse.h"
#include <JavaScriptCore/TypedArrays.h>
#include <span>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>

namespace JSC {
class CallFrame;
class JSValue;
};

namespace WebCore {

class AbortSignal;
class FetchRequest;
class FetchResponseBodyLoader;
class ReadableStreamSource;

class FetchResponse final : public FetchBodyOwner {
public:
    using Type = ResourceResponse::Type;

    struct Init {
        unsigned short status { httpStatus200OK };
        AtomString statusText;
        std::optional<FetchHeaders::Init> headers;
    };

    virtual ~FetchResponse();

    WEBCORE_EXPORT static Ref<FetchResponse> create(ScriptExecutionContext*, std::optional<FetchBody>&&, FetchHeaders::Guard, ResourceResponse&&);

    static ExceptionOr<Ref<FetchResponse>> create(ScriptExecutionContext&, std::optional<FetchBody::Init>&&, Init&&);
    static ExceptionOr<Ref<FetchResponse>> create(ScriptExecutionContext&, std::optional<FetchBodyWithType>&&, Init&&);
    static Ref<FetchResponse> error(ScriptExecutionContext&);
    static ExceptionOr<Ref<FetchResponse>> redirect(ScriptExecutionContext&, const String& url, int status);
    static ExceptionOr<Ref<FetchResponse>> jsonForBindings(ScriptExecutionContext&, JSC::JSValue data, Init&&);

    using NotificationCallback = Function<void(ExceptionOr<Ref<FetchResponse>>&&)>;
    static void fetch(ScriptExecutionContext&, FetchRequest&, NotificationCallback&&, const String& initiator);
    static Ref<FetchResponse> createFetchResponse(ScriptExecutionContext&, FetchRequest&, NotificationCallback&&);

    void startConsumingStream(unsigned);
    void consumeChunk(Ref<JSC::Uint8Array>&&);
    void finishConsumingStream(Ref<DeferredPromise>&&);

    Type type() const { return filteredResponse().type(); }
    const String& url() const;
    bool redirected() const { return filteredResponse().isRedirected(); }
    int status() const { return filteredResponse().httpStatusCode(); }
    bool ok() const { return filteredResponse().isSuccessful(); }
    const String& statusText() const { return filteredResponse().httpStatusText(); }

    const FetchHeaders& headers() const { return m_headers; }
    FetchHeaders& headers() { return m_headers; }
    ExceptionOr<Ref<FetchResponse>> clone();

    void consumeBodyAsStream() final;
    void feedStream() final;
    void cancel() final;

    using ResponseData = std::variant<std::nullptr_t, Ref<FormData>, Ref<SharedBuffer>>;
    ResponseData consumeBody();
    void setBodyData(ResponseData&&, uint64_t bodySizeWithPadding);

    bool isLoading() const { return !!m_loader; }
    bool isBodyReceivedByChunk() const { return isLoading() || hasReadableStreamBody(); }
    bool isBlobBody() const { return !isBodyNull() && body().isBlob(); }
    bool isBlobFormData() const { return !isBodyNull() && body().isFormData(); }

    using ConsumeDataByChunkCallback = Function<void(ExceptionOr<std::span<const uint8_t>*>&&)>;
    void consumeBodyReceivedByChunk(ConsumeDataByChunkCallback&&);
    void cancelStream();

    WEBCORE_EXPORT ResourceResponse resourceResponse() const;
    ResourceResponse::Tainting tainting() const { return m_internalResponse.tainting(); }

    uint64_t bodySizeWithPadding() const { return m_bodySizeWithPadding; }
    void setBodySizeWithPadding(uint64_t size) { m_bodySizeWithPadding = size; }
    uint64_t opaqueLoadIdentifier() const { return m_opaqueLoadIdentifier; }

    void initializeOpaqueLoadIdentifierForTesting() { m_opaqueLoadIdentifier = 1; }

    const HTTPHeaderMap& internalResponseHeaders() const { return m_internalResponse.httpHeaderFields(); }

    bool isCORSSameOrigin() const;
    bool hasWasmMIMEType() const;

    const NetworkLoadMetrics& networkLoadMetrics() const { return m_networkLoadMetrics; }
    void setReceivedInternalResponse(const ResourceResponse&, FetchOptions::Credentials);
    void startLoader(ScriptExecutionContext&, FetchRequest&, const String& initiator);

    void setIsNavigationPreload(bool isNavigationPreload) { m_isNavigationPreload = isNavigationPreload; }
    bool isAvailableNavigationPreload() const { return m_isNavigationPreload && m_loader && !m_loader->hasLoader() && !hasReadableStreamBody(); }
    void markAsUsedForPreload();
    bool isUsedForPreload() const { return m_isUsedForPreload; }

    void setBodyLoader(UniqueRef<FetchResponseBodyLoader>&&);
    void receivedError(Exception&&);
    void receivedError(ResourceError&&);
    void didSucceed(const NetworkLoadMetrics&);
    void receivedData(Ref<SharedBuffer>&&);

private:
    FetchResponse(ScriptExecutionContext*, std::optional<FetchBody>&&, Ref<FetchHeaders>&&, ResourceResponse&&);

    // FetchBodyOwner
    void stop() final;
    void loadBody() final;

    const ResourceResponse& filteredResponse() const;
    void setNetworkLoadMetrics(const NetworkLoadMetrics& metrics) { m_networkLoadMetrics = metrics; }
    void closeStream();

    void addAbortSteps(Ref<AbortSignal>&&);
    void processReceivedError();

    class Loader final : public FetchLoaderClient {
        WTF_MAKE_TZONE_ALLOCATED(Loader);
        WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(Loader);
    public:
        Loader(FetchResponse&, NotificationCallback&&);
        ~Loader();

        bool start(ScriptExecutionContext&, const FetchRequest&, const String& initiator);
        void stop();

        void consumeDataByChunk(ConsumeDataByChunkCallback&&);

        bool hasLoader() const { return !!m_loader; }

        RefPtr<FragmentedSharedBuffer> startStreaming();
        NotificationCallback takeNotificationCallback() { return WTFMove(m_responseCallback); }
        ConsumeDataByChunkCallback takeConsumeDataCallback() { return WTFMove(m_consumeDataCallback); }

    private:
        // FetchLoaderClient API
        void didSucceed(const NetworkLoadMetrics&) final;
        void didFail(const ResourceError&) final;
        void didReceiveResponse(const ResourceResponse&) final;
        void didReceiveData(const SharedBuffer&) final;

        WeakRef<FetchResponse> m_response;
        NotificationCallback m_responseCallback;
        ConsumeDataByChunkCallback m_consumeDataCallback;
        std::unique_ptr<FetchLoader> m_loader;
        Ref<PendingActivity<FetchResponse>> m_pendingActivity;
        FetchOptions::Credentials m_credentials;
        bool m_shouldStartStreaming { false };
    };

    mutable std::optional<ResourceResponse> m_filteredResponse;
    ResourceResponse m_internalResponse;
    std::unique_ptr<Loader> m_loader;
    std::unique_ptr<FetchResponseBodyLoader> m_bodyLoader;
    mutable String m_responseURL;
    // Opaque responses will padd their body size when used with Cache API.
    uint64_t m_bodySizeWithPadding { 0 };
    uint64_t m_opaqueLoadIdentifier { 0 };
    RefPtr<AbortSignal> m_abortSignal;
    NetworkLoadMetrics m_networkLoadMetrics;
    bool m_hasInitializedInternalResponse { false };
    bool m_isNavigationPreload { false };
    bool m_isUsedForPreload { false };
};

} // namespace WebCore
