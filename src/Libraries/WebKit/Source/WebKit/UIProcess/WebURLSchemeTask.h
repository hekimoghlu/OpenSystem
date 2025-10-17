/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 23, 2022.
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

#include "APIObject.h"
#include "WebProcessProxy.h"
#include <WebCore/ResourceRequest.h>
#include <WebCore/ResourceResponse.h>
#include <WebCore/SharedBuffer.h>
#include <wtf/CompletionHandler.h>
#include <wtf/InstanceCounted.h>
#include <wtf/Lock.h>
#include <wtf/RefCounted.h>
#include <wtf/RefPtr.h>

namespace API {
class FrameInfo;
}

namespace WebCore {
class ResourceError;
class ResourceResponse;
}

namespace WebKit {

struct URLSchemeTaskParameters;
class WebURLSchemeHandler;
class WebPageProxy;

using SyncLoadCompletionHandler = CompletionHandler<void(const WebCore::ResourceResponse&, const WebCore::ResourceError&, Vector<uint8_t>&&)>;

class WebURLSchemeTask : public API::ObjectImpl<API::Object::Type::URLSchemeTask>, public InstanceCounted<WebURLSchemeTask> {
    WTF_MAKE_NONCOPYABLE(WebURLSchemeTask);
public:
    static Ref<WebURLSchemeTask> create(WebURLSchemeHandler&, WebPageProxy&, WebProcessProxy&, WebCore::PageIdentifier, URLSchemeTaskParameters&&, SyncLoadCompletionHandler&&);

    ~WebURLSchemeTask();

    WebCore::ResourceLoaderIdentifier resourceLoaderID() const { ASSERT(RunLoop::isMain()); return m_resourceLoaderID; }
    std::optional<WebPageProxyIdentifier> pageProxyID() const { ASSERT(RunLoop::isMain()); return m_pageProxyID; }
    std::optional<WebCore::PageIdentifier> webPageID() const { ASSERT(RunLoop::isMain()); return m_webPageID.asOptional(); }
    WebProcessProxy* process() { ASSERT(RunLoop::isMain()); return m_process.get(); }
    WebCore::ResourceRequest request() const;
    API::FrameInfo& frameInfo() const { return m_frameInfo.get(); }

#if PLATFORM(COCOA)
    NSURLRequest *nsRequest() const;
#endif

    enum class ExceptionType {
        DataAlreadySent,
        CompleteAlreadyCalled,
        RedirectAfterResponse,
        TaskAlreadyStopped,
        NoResponseSent,
        WaitingForRedirectCompletionHandler,
        None,
    };
    ExceptionType willPerformRedirection(WebCore::ResourceResponse&&, WebCore::ResourceRequest&&,  Function<void(WebCore::ResourceRequest&&)>&&);
    ExceptionType didPerformRedirection(WebCore::ResourceResponse&&, WebCore::ResourceRequest&&);
    ExceptionType didReceiveResponse(const WebCore::ResourceResponse&);
    ExceptionType didReceiveData(Ref<WebCore::SharedBuffer>&&);
    ExceptionType didComplete(const WebCore::ResourceError&);

    void stop();
    void pageDestroyed();

    void suppressTaskStoppedExceptions() { m_shouldSuppressTaskStoppedExceptions = true; }

    bool waitingForRedirectCompletionHandlerCallback() const { return m_waitingForRedirectCompletionHandlerCallback; }

private:
    WebURLSchemeTask(WebURLSchemeHandler&, WebPageProxy&, WebProcessProxy&, WebCore::PageIdentifier, URLSchemeTaskParameters&&, SyncLoadCompletionHandler&&);

    bool isSync() const { return !!m_syncCompletionHandler; }

    Ref<WebURLSchemeHandler> m_urlSchemeHandler;
    RefPtr<WebProcessProxy> m_process;
    WebCore::ResourceLoaderIdentifier m_resourceLoaderID;
    Markable<WebPageProxyIdentifier> m_pageProxyID;
    Markable<WebCore::PageIdentifier> m_webPageID;
    WebCore::ResourceRequest m_request WTF_GUARDED_BY_LOCK(m_requestLock);
    Ref<API::FrameInfo> m_frameInfo;
    mutable Lock m_requestLock;
    bool m_stopped { false };
    bool m_responseSent { false };
    bool m_dataSent { false };
    bool m_completed { false };
    bool m_shouldSuppressTaskStoppedExceptions { false };

    SyncLoadCompletionHandler m_syncCompletionHandler;
    WebCore::ResourceResponse m_syncResponse;
    WebCore::SharedBufferBuilder m_syncData;

    bool m_waitingForRedirectCompletionHandlerCallback { false };
};

} // namespace WebKit
