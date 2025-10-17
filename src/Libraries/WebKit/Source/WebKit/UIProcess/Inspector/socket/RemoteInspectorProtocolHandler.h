/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 21, 2024.
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
#include "WebURLSchemeHandler.h"

namespace WTF {
class URL;
}

namespace WebKit {

class WebURLSchemeTask;

class RemoteInspectorProtocolHandler final : public RemoteInspectorObserver, public WebURLSchemeHandler {
    WTF_MAKE_FAST_ALLOCATED;
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(RemoteInspectorProtocolHandler);
public:
    static Ref<RemoteInspectorProtocolHandler> create(WebPageProxy& page) { return adoptRef(*new RemoteInspectorProtocolHandler(page)); }

    void inspect(const String& hostAndPort, ConnectionID, TargetID, const String& type);

private:
    RemoteInspectorProtocolHandler(WebPageProxy& page)
        : m_page(page) { }

    // RemoteInspectorObserver
    void targetListChanged(RemoteInspectorClient&) final;
    void connectionClosed(RemoteInspectorClient&) final { }

    // WebURLSchemeHandler
    void platformStartTask(WebPageProxy&, WebURLSchemeTask&) final;
    void platformStopTask(WebPageProxy&, WebURLSchemeTask&) final { }

    void updateTargetList();

    void runScript(const String&);
    Ref<WebPageProxy> protectedPage() const;

    std::unique_ptr<RemoteInspectorClient> m_inspectorClient;
    WeakRef<WebPageProxy> m_page;
    bool m_pageLoaded { false };
    String m_targetListsHtml;
};

} // namespace WebKit

#endif // ENABLE(REMOTE_INSPECTOR)
