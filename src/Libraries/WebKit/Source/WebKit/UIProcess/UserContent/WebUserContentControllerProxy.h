/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 5, 2022.
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
#include "ContentWorldShared.h"
#include "MessageReceiver.h"
#include "ScriptMessageHandlerIdentifier.h"
#include "UserContentControllerIdentifier.h"
#include "WebPageProxyIdentifier.h"
#include "WebUserContentControllerProxyMessages.h"
#include <WebCore/PageIdentifier.h>
#include <wtf/CheckedRef.h>
#include <wtf/Forward.h>
#include <wtf/HashCountedSet.h>
#include <wtf/HashMap.h>
#include <wtf/Identified.h>
#include <wtf/Ref.h>
#include <wtf/RefCounted.h>
#include <wtf/URL.h>
#include <wtf/URLHash.h>
#include <wtf/WeakHashSet.h>
#include <wtf/text/StringHash.h>

namespace API {
class Array;
class ContentRuleList;
class ContentWorld;
class UserScript;
class UserStyleSheet;
}

namespace WebCore {
class SecurityOriginData;
}

namespace WebKit {

class NetworkProcessProxy;
class WebProcessProxy;
class WebScriptMessageHandler;
struct FrameInfoData;
class WebCompiledContentRuleListData;
struct WebPageCreationParameters;
struct UserContentControllerParameters;
enum class InjectUserScriptImmediately : bool;

class WebUserContentControllerProxy : public API::ObjectImpl<API::Object::Type::UserContentController>, public IPC::MessageReceiver, public Identified<UserContentControllerIdentifier> {
public:
#if ENABLE(WK_WEB_EXTENSIONS)
    enum class RemoveWebExtensions : bool { No, Yes };
#endif

    static Ref<WebUserContentControllerProxy> create()
    {
        return adoptRef(*new WebUserContentControllerProxy);
    }

    WebUserContentControllerProxy();
    ~WebUserContentControllerProxy();

    void ref() const final { API::ObjectImpl<API::Object::Type::UserContentController>::ref(); }
    void deref() const final { API::ObjectImpl<API::Object::Type::UserContentController>::deref(); }

    static WebUserContentControllerProxy* get(UserContentControllerIdentifier);

    UserContentControllerParameters parameters() const;

    void addProcess(WebProcessProxy&);
    void removeProcess(WebProcessProxy&);

    API::Array& userScripts() { return m_userScripts.get(); }
    Ref<API::Array> protectedUserScripts();
    void addUserScript(API::UserScript&, InjectUserScriptImmediately);
    void removeUserScript(API::UserScript&);
    void removeAllUserScripts(API::ContentWorld&);
#if ENABLE(WK_WEB_EXTENSIONS)
    void removeAllUserScripts(RemoveWebExtensions = RemoveWebExtensions::No);
#else
    void removeAllUserScripts();
#endif

    API::Array& userStyleSheets() { return m_userStyleSheets.get(); }
    void addUserStyleSheet(API::UserStyleSheet&);
    void removeUserStyleSheet(API::UserStyleSheet&);
    void removeAllUserStyleSheets(API::ContentWorld&);
#if ENABLE(WK_WEB_EXTENSIONS)
    void removeAllUserStyleSheets(RemoveWebExtensions = RemoveWebExtensions::No);
#else
    void removeAllUserStyleSheets();
#endif

    // Returns false if there was a name conflict.
    bool addUserScriptMessageHandler(WebScriptMessageHandler&);
    void removeUserMessageHandlerForName(const String&, API::ContentWorld&);
    void removeAllUserMessageHandlers(API::ContentWorld&);
    void removeAllUserMessageHandlers();

#if ENABLE(CONTENT_EXTENSIONS)
    void addNetworkProcess(NetworkProcessProxy&);
    void removeNetworkProcess(NetworkProcessProxy&);

    void addContentRuleList(API::ContentRuleList&, const WTF::URL& extensionBaseURL = { });
    void removeContentRuleList(const String&);
#if ENABLE(WK_WEB_EXTENSIONS)
    void removeAllContentRuleLists(RemoveWebExtensions = RemoveWebExtensions::No);
#else
    void removeAllContentRuleLists();
#endif

    const HashMap<String, std::pair<Ref<API::ContentRuleList>, URL>>& contentExtensionRules() { return m_contentRuleLists; }
    Vector<std::pair<WebCompiledContentRuleListData, URL>> contentRuleListData() const;
#endif

    void contentWorldDestroyed(API::ContentWorld&);

    bool operator==(const WebUserContentControllerProxy& other) const { return (this == &other); }

private:
    Ref<API::Array> protectedUserScripts() const;
    Ref<API::Array> protectedUserStyleSheets() const;

    // IPC::MessageReceiver.
    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) override;

    void didPostMessage(WebPageProxyIdentifier, FrameInfoData&&, ScriptMessageHandlerIdentifier, std::span<const uint8_t>, CompletionHandler<void(std::span<const uint8_t>, const String&)>&&);

    void addContentWorld(API::ContentWorld&);

    WeakHashSet<WebProcessProxy> m_processes;
    Ref<API::Array> m_userScripts;
    Ref<API::Array> m_userStyleSheets;
    HashMap<ScriptMessageHandlerIdentifier, RefPtr<WebScriptMessageHandler>> m_scriptMessageHandlers;
    HashSet<ContentWorldIdentifier> m_associatedContentWorlds;

#if ENABLE(CONTENT_EXTENSIONS)
    WeakHashSet<NetworkProcessProxy> m_networkProcesses;
    HashMap<String, std::pair<Ref<API::ContentRuleList>, URL>> m_contentRuleLists;
#endif
};

} // namespace WebKit
