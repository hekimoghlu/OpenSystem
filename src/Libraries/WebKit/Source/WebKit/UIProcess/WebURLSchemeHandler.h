/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 8, 2021.
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
#include "WebURLSchemeTask.h"
#include <WebCore/ResourceLoaderIdentifier.h>
#include <wtf/HashMap.h>
#include <wtf/HashSet.h>
#include <wtf/Identified.h>
#include <wtf/RefCounted.h>
#include <wtf/RefPtr.h>

namespace WebCore {
class ResourceRequest;
}

namespace WebKit {

struct URLSchemeTaskParameters;
class WebPageProxy;
class WebProcessProxy;

using SyncLoadCompletionHandler = CompletionHandler<void(const WebCore::ResourceResponse&, const WebCore::ResourceError&, Vector<uint8_t>&&)>;

class WebURLSchemeHandler : public RefCounted<WebURLSchemeHandler>, public Identified<WebURLSchemeHandlerIdentifier> {
    WTF_MAKE_NONCOPYABLE(WebURLSchemeHandler);
public:
    virtual ~WebURLSchemeHandler();

    void startTask(WebPageProxy&, WebProcessProxy&, WebCore::PageIdentifier, URLSchemeTaskParameters&&, SyncLoadCompletionHandler&&);
    void stopTask(WebPageProxy&, WebCore::ResourceLoaderIdentifier taskIdentifier);
    void stopAllTasksForPage(WebPageProxy&, WebProcessProxy*);
    void taskCompleted(WebPageProxyIdentifier, WebURLSchemeTask&);

    virtual bool isAPIHandler() { return false; }

protected:
    WebURLSchemeHandler();

private:
    virtual void platformStartTask(WebPageProxy&, WebURLSchemeTask&) = 0;
    virtual void platformStopTask(WebPageProxy&, WebURLSchemeTask&) = 0;
    virtual void platformTaskCompleted(WebURLSchemeTask&) { };

    void removeTaskFromPageMap(WebPageProxyIdentifier, WebCore::ResourceLoaderIdentifier);
    WebProcessProxy* processForTaskIdentifier(WebPageProxy&, WebCore::ResourceLoaderIdentifier) const;

    HashMap<std::pair<WebCore::ResourceLoaderIdentifier, WebPageProxyIdentifier>, Ref<WebURLSchemeTask>> m_tasks;
    HashMap<WebPageProxyIdentifier, HashSet<WebCore::ResourceLoaderIdentifier>> m_tasksByPageIdentifier;
    
    SyncLoadCompletionHandler m_syncLoadCompletionHandler;

}; // class WebURLSchemeHandler

} // namespace WebKit
