/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 3, 2023.
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

#include "WebURLSchemeHandlerIdentifier.h"
#include "WebURLSchemeTaskProxy.h"
#include <wtf/CheckedRef.h>
#include <wtf/HashMap.h>
#include <wtf/RefCountedAndCanMakeWeakPtr.h>

namespace WebCore {
class ResourceError;
class ResourceLoader;
class ResourceResponse;
class ResourceRequest;
class SharedBuffer;
}

namespace WebKit {

class WebPage;

class WebURLSchemeHandlerProxy : public RefCountedAndCanMakeWeakPtr<WebURLSchemeHandlerProxy> {
public:
    static Ref<WebURLSchemeHandlerProxy> create(WebPage& page, WebURLSchemeHandlerIdentifier identifier)
    {
        return adoptRef(*new WebURLSchemeHandlerProxy(page, identifier));
    }
    ~WebURLSchemeHandlerProxy();

    void startNewTask(WebCore::ResourceLoader&, WebFrame&);
    void stopAllTasks();

    void loadSynchronously(WebCore::ResourceLoaderIdentifier, WebFrame&, const WebCore::ResourceRequest&, WebCore::ResourceResponse&, WebCore::ResourceError&, Vector<uint8_t>&);

    WebURLSchemeHandlerIdentifier identifier() const { return m_identifier; }
    WebPage& page() { return m_webPage.get(); }
    Ref<WebPage> protectedPage();

    void taskDidPerformRedirection(WebCore::ResourceLoaderIdentifier, WebCore::ResourceResponse&&, WebCore::ResourceRequest&&, CompletionHandler<void(WebCore::ResourceRequest&&)>&&);
    void taskDidReceiveResponse(WebCore::ResourceLoaderIdentifier, const WebCore::ResourceResponse&);
    void taskDidReceiveData(WebCore::ResourceLoaderIdentifier, Ref<WebCore::SharedBuffer>&&);
    void taskDidComplete(WebCore::ResourceLoaderIdentifier, const WebCore::ResourceError&);
    void taskDidStopLoading(WebURLSchemeTaskProxy&);

private:
    WebURLSchemeHandlerProxy(WebPage&, WebURLSchemeHandlerIdentifier);

    RefPtr<WebURLSchemeTaskProxy> removeTask(WebCore::ResourceLoaderIdentifier);

    WeakRef<WebPage> m_webPage;
    WebURLSchemeHandlerIdentifier m_identifier;

    HashMap<WebCore::ResourceLoaderIdentifier, RefPtr<WebURLSchemeTaskProxy>> m_tasks;
}; // class WebURLSchemeHandlerProxy

} // namespace WebKit
