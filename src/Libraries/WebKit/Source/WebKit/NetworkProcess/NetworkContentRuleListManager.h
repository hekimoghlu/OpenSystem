/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 28, 2023.
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

#if ENABLE(CONTENT_EXTENSIONS)

#include "UserContentControllerIdentifier.h"
#include "WebCompiledContentRuleListData.h"
#include <WebCore/ContentExtensionsBackend.h>
#include <wtf/WeakRef.h>

namespace IPC {
class Connection;
class Decoder;
}

namespace WebKit {

class NetworkProcess;

class NetworkContentRuleListManager {
public:
    NetworkContentRuleListManager(NetworkProcess&);
    ~NetworkContentRuleListManager();

    void didReceiveMessage(IPC::Connection&, IPC::Decoder&);

    using BackendCallback = CompletionHandler<void(WebCore::ContentExtensions::ContentExtensionsBackend&)>;
    void contentExtensionsBackend(UserContentControllerIdentifier, BackendCallback&&);

    void ref() const;
    void deref() const;

private:
    void addContentRuleLists(UserContentControllerIdentifier, Vector<std::pair<WebCompiledContentRuleListData, URL>>&&);
    void removeContentRuleList(UserContentControllerIdentifier, const String& name);
    void removeAllContentRuleLists(UserContentControllerIdentifier);
    void remove(UserContentControllerIdentifier);

    Ref<NetworkProcess> protectedNetworkProcess() const;

    HashMap<UserContentControllerIdentifier, std::unique_ptr<WebCore::ContentExtensions::ContentExtensionsBackend>> m_contentExtensionBackends;
    HashMap<UserContentControllerIdentifier, Vector<BackendCallback>> m_pendingCallbacks;
    WeakRef<NetworkProcess> m_networkProcess;
};

} // namespace WebKit

#endif // ENABLE(CONTENT_EXTENSIONS)
