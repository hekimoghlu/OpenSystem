/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 15, 2021.
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

#include "ActiveDOMObject.h"
#include "ExceptionOr.h"
#include "FormData.h"
#include "ResourceResponse.h"
#include "SharedBuffer.h"
#include "ThreadableLoaderClient.h"
#include "Timer.h"
#include "URLKeepingBlobAlive.h"
#include "UserGestureIndicator.h"
#include <wtf/URL.h>
#include "XMLHttpRequestEventTarget.h"
#include "XMLHttpRequestProgressEventThrottle.h"
#include <variant>
#include <wtf/CancellableTask.h>
#include <wtf/text/StringBuilder.h>

namespace JSC {
class ArrayBuffer;
class ArrayBufferView;
}

namespace WebCore {

class Blob;
class Document;
class DOMFormData;
class SecurityOrigin;
class TextResourceDecoder;
class ThreadableLoader;
class URLSearchParams;
class XMLHttpRequestUpload;
struct OwnedString;

class XMLHttpRequest final : public ActiveDOMObject, public RefCounted<XMLHttpRequest>, private ThreadableLoaderClient, public XMLHttpRequestEventTarget {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED_EXPORT(XMLHttpRequest, WEBCORE_EXPORT);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(XMLHttpRequest);
public:
    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    static Ref<XMLHttpRequest> create(ScriptExecutionContext&);
    WEBCORE_EXPORT ~XMLHttpRequest();

    // Keep it in 3bits.
    enum State : uint8_t {
        UNSENT = 0,
        OPENED = 1,
        HEADERS_RECEIVED = 2,
        LOADING = 3,
        DONE = 4
    };

    virtual void didReachTimeout();

    enum EventTargetInterfaceType eventTargetInterface() const override { return EventTargetInterfaceType::XMLHttpRequest; }
    ScriptExecutionContext* scriptExecutionContext() const override { return ActiveDOMObject::scriptExecutionContext(); }

    using SendTypes = std::variant<RefPtr<Document>, RefPtr<Blob>, RefPtr<JSC::ArrayBufferView>, RefPtr<JSC::ArrayBuffer>, RefPtr<DOMFormData>, String, RefPtr<URLSearchParams>>;

    const URL& url() const { return m_url; }
    String statusText() const;
    int status() const;
    State readyState() const { return static_cast<State>(m_readyState); }
    bool withCredentials() const { return m_includeCredentials; }
    ExceptionOr<void> setWithCredentials(bool);
    ExceptionOr<void> open(const String& method, const String& url);
    ExceptionOr<void> open(const String& method, const URL&, bool async);
    ExceptionOr<void> open(const String& method, const String&, bool async, const String& user, const String& password);
    ExceptionOr<void> send(std::optional<SendTypes>&&);
    void abort();
    ExceptionOr<void> setRequestHeader(const String& name, const String& value);
    ExceptionOr<void> overrideMimeType(const String& override);
    bool doneWithoutErrors() const { return !m_error && readyState() == DONE; }
    String getAllResponseHeaders() const;
    String getResponseHeader(const String& name) const;
    ExceptionOr<OwnedString> responseText();
    String responseTextIgnoringResponseType() const { return m_responseBuilder.toStringPreserveCapacity(); }
    enum class FinalMIMEType : bool { No, Yes };
    String responseMIMEType(FinalMIMEType = FinalMIMEType::No) const;

    Document* optionalResponseXML() const { return m_responseDocument.get(); }
    ExceptionOr<Document*> responseXML();

    Ref<Blob> createResponseBlob();
    RefPtr<JSC::ArrayBuffer> createResponseArrayBuffer();

    unsigned timeout() const { return m_timeoutMilliseconds; }
    ExceptionOr<void> setTimeout(unsigned);

    bool responseCacheIsValid() const { return m_responseCacheIsValid; }
    void didCacheResponse();

    // Keep it in 3bits.
    enum class ResponseType : uint8_t {
        EmptyString = 0,
        Arraybuffer = 1,
        Blob = 2,
        Document = 3,
        Json = 4,
        Text = 5,
    };
    ExceptionOr<void> setResponseType(ResponseType);
    ResponseType responseType() const { return static_cast<ResponseType>(m_responseType); }

    String responseURL() const;

    XMLHttpRequestUpload& upload();
    XMLHttpRequestUpload* optionalUpload() const { return m_upload.get(); }

    const ResourceResponse& resourceResponse() const { return m_response; }

    size_t memoryCost() const;

    using EventTarget::dispatchEvent;
    void dispatchEvent(Event&) override;

private:
    friend class XMLHttpRequestUpload;
    explicit XMLHttpRequest(ScriptExecutionContext&);

    void updateHasRelevantEventListener();
    void handleCancellation();

    // EventTarget.
    void eventListenersDidChange() final;

    PAL::TextEncoding finalResponseCharset() const;

    // ActiveDOMObject
    void contextDestroyed() override;
    void suspend(ReasonForSuspension) override;
    void resume() override;
    void stop() override;
    bool virtualHasPendingActivity() const final;

    void refEventTarget() override { ref(); }
    void derefEventTarget() override { deref(); }

    Document* document() const;
    SecurityOrigin* securityOrigin() const;

    // ThreadableLoaderClient
    void didSendData(unsigned long long bytesSent, unsigned long long totalBytesToBeSent) override;
    void didReceiveResponse(ScriptExecutionContextIdentifier, std::optional<ResourceLoaderIdentifier>, const ResourceResponse&) override;
    void didReceiveData(const SharedBuffer&) override;
    void didFinishLoading(ScriptExecutionContextIdentifier, std::optional<ResourceLoaderIdentifier>, const NetworkLoadMetrics&) override;
    void didFail(std::optional<ScriptExecutionContextIdentifier>, const ResourceError&) override;
    void notifyIsDone(bool) final;

    std::optional<ExceptionOr<void>> prepareToSend();
    ExceptionOr<void> send(const URLSearchParams&);
    ExceptionOr<void> send(Document&);
    ExceptionOr<void> send(const String& = { });
    ExceptionOr<void> send(Blob&);
    ExceptionOr<void> send(DOMFormData&);
    ExceptionOr<void> send(JSC::ArrayBuffer&);
    ExceptionOr<void> send(JSC::ArrayBufferView&);
    ExceptionOr<void> sendBytesData(std::span<const uint8_t>);

    void changeState(State);
    void callReadyStateChangeListener();

    // Returns false when cancelling the loader within internalAbort() triggers an event whose callback creates a new loader. 
    // In that case, the function calling internalAbort should exit.
    bool internalAbort();

    void clearResponse();
    void clearResponseBuffers();
    void clearRequest();

    ExceptionOr<void> createRequest();

    void timeoutTimerFired();

    void genericError();
    void networkError();
    void abortError();

    void dispatchErrorEvents(const AtomString&);

    Ref<TextResourceDecoder> createDecoder() const;

    unsigned m_async : 1;
    unsigned m_includeCredentials : 1;
    unsigned m_sendFlag : 1;
    unsigned m_createdDocument : 1;
    unsigned m_error : 1;
    unsigned m_uploadListenerFlag : 1;
    unsigned m_uploadComplete : 1;
    unsigned m_responseCacheIsValid : 1;
    unsigned m_readyState : 3; // State
    unsigned m_responseType : 3; // ResponseType

    unsigned m_timeoutMilliseconds { 0 };

    std::unique_ptr<XMLHttpRequestUpload> m_upload;

    URLKeepingBlobAlive m_url;
    String m_method;
    HTTPHeaderMap m_requestHeaders;
    RefPtr<FormData> m_requestEntityBody;
    String m_mimeTypeOverride;

    struct LoadingActivity {
        Ref<XMLHttpRequest> protectedThis; // Keep object alive while loading even if there is no longer a JS wrapper.
        Ref<ThreadableLoader> loader;
    };
    std::optional<LoadingActivity> m_loadingActivity;

    String m_responseEncoding;

    ResourceResponse m_response;

    RefPtr<TextResourceDecoder> m_decoder;

    RefPtr<Document> m_responseDocument;

    SharedBufferBuilder m_binaryResponseBuilder;

    StringBuilder m_responseBuilder;

    // Used for progress event tracking.
    long long m_receivedLength { 0 };

    XMLHttpRequestProgressEventThrottle m_progressEventThrottle;

    mutable String m_allResponseHeaders;

    Timer m_timeoutTimer;

    MonotonicTime m_sendingTime;

    std::optional<ExceptionCode> m_exceptionCode;
    RefPtr<UserGestureToken> m_userGestureToken;
    bool m_hasRelevantEventListener { false };
    TaskCancellationGroup m_abortErrorGroup;
    bool m_wasDidSendDataCalledForTotalBytes { false };
};

} // namespace WebCore
