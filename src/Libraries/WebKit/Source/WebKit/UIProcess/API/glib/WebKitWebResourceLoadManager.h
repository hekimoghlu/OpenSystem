/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 30, 2022.
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

#include "WebKitWebResource.h"
#include "WebKitWebView.h"
#include <WebCore/GlobalFrameIdentifier.h>
#include <WebCore/ResourceLoaderIdentifier.h>
#include <wtf/HashMap.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/glib/GRefPtr.h>

namespace WebCore {
class ResourceError;
class ResourceRequest;
class ResourceResponse;
}

namespace WebKit {

class WebKitWebResourceLoadManager {
    WTF_MAKE_TZONE_ALLOCATED(WebKitWebResourceLoadManager);
public:
    explicit WebKitWebResourceLoadManager(WebKitWebView*);
    ~WebKitWebResourceLoadManager();

    void didInitiateLoad(WebCore::ResourceLoaderIdentifier, WebCore::FrameIdentifier, WebCore::ResourceRequest&&);
    void didSendRequest(WebCore::ResourceLoaderIdentifier, WebCore::FrameIdentifier, WebCore::ResourceRequest&&, WebCore::ResourceResponse&&);
    void didReceiveResponse(WebCore::ResourceLoaderIdentifier, WebCore::FrameIdentifier, WebCore::ResourceResponse&&);
    void didFinishLoad(WebCore::ResourceLoaderIdentifier, WebCore::FrameIdentifier, WebCore::ResourceError&&);

private:
    WebKitWebView* m_webView { nullptr };
    HashMap<std::pair<WebCore::ResourceLoaderIdentifier, WebCore::FrameIdentifier>, GRefPtr<WebKitWebResource>> m_resources;
};

} // namespace WebKit
