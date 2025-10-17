/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 9, 2025.
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

#if ENABLE(REMOTE_INSPECTOR)

#include "RemoteInspectorClient.h"
#include "WebKitURISchemeRequest.h"
#include "WebKitUserContentManager.h"
#include "WebKitWebView.h"
#include <wtf/HashMap.h>
#include <wtf/HashSet.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/text/StringHash.h>

namespace WebKit {

class RemoteInspectorProtocolHandler final : public RemoteInspectorObserver {
    WTF_MAKE_TZONE_ALLOCATED(RemoteInspectorProtocolHandler);
public:
    explicit RemoteInspectorProtocolHandler(WebKitWebContext* context);
    ~RemoteInspectorProtocolHandler();

    void inspect(const String& hostAndPort, uint64_t connectionID, uint64_t targetID, const String& targetType);

private:
    static void webViewDestroyed(RemoteInspectorProtocolHandler*, WebKitWebView*);
    static void userContentManagerDestroyed(RemoteInspectorProtocolHandler*, WebKitUserContentManager*);

    void handleRequest(WebKitURISchemeRequest*);
    void updateTargetList(WebKitWebView*);
    static void webViewLoadChanged(WebKitWebView*, WebKitLoadEvent, RemoteInspectorProtocolHandler*);

    // RemoteInspectorObserver.
    void targetListChanged(RemoteInspectorClient&) override;
    void connectionClosed(RemoteInspectorClient&) override;

    HashMap<String, std::unique_ptr<RemoteInspectorClient>> m_inspectorClients;
    HashSet<WebKitUserContentManager*> m_userContentManagers;
    HashMap<WebKitWebView*, RemoteInspectorClient*> m_webViews;
};

} // namespace WebKit

#endif // ENABLE(REMOTE_INSPECTOR)
