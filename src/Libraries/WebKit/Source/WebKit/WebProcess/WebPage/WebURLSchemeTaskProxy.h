/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 20, 2025.
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

#include <WebCore/ResourceLoaderIdentifier.h>
#include <WebCore/ResourceRequest.h>
#include <wtf/Deque.h>
#include <wtf/RefCountedAndCanMakeWeakPtr.h>

namespace WebCore {
class ResourceError;
class ResourceLoader;
class ResourceResponse;
class SharedBuffer;
}

namespace WebKit {

class WebFrame;
class WebURLSchemeHandlerProxy;

class WebURLSchemeTaskProxy : public RefCountedAndCanMakeWeakPtr<WebURLSchemeTaskProxy> {
public:
    static Ref<WebURLSchemeTaskProxy> create(WebURLSchemeHandlerProxy& handler, WebCore::ResourceLoader& loader, WebFrame& webFrame)
    {
        return adoptRef(*new WebURLSchemeTaskProxy(handler, loader, webFrame));
    }
    
    const WebCore::ResourceRequest& request() const { return m_request; }

    void startLoading();
    void stopLoading();

    void didPerformRedirection(WebCore::ResourceResponse&&, WebCore::ResourceRequest&&, CompletionHandler<void(WebCore::ResourceRequest&&)>&&);
    void didReceiveResponse(const WebCore::ResourceResponse&);
    void didReceiveData(const WebCore::SharedBuffer&);
    void didComplete(const WebCore::ResourceError&);

    WebCore::ResourceLoaderIdentifier identifier() const { return m_identifier; }

private:
    WebURLSchemeTaskProxy(WebURLSchemeHandlerProxy&, WebCore::ResourceLoader&, WebFrame&);
    bool hasLoader();

    void queueTask(Function<void()>&& task) { m_queuedTasks.append(WTFMove(task)); }
    void processNextPendingTask();

    WeakRef<WebURLSchemeHandlerProxy> m_urlSchemeHandler;
    RefPtr<WebCore::ResourceLoader> m_coreLoader;
    RefPtr<WebFrame> m_frame;
    WebCore::ResourceRequest m_request;
    WebCore::ResourceLoaderIdentifier m_identifier;
    bool m_waitingForCompletionHandler { false };
    Deque<Function<void()>> m_queuedTasks;
};

} // namespace WebKit
