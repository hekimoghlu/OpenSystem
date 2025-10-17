/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 29, 2023.
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

#include "MessageReceiver.h"
#include "ScriptMessageHandlerIdentifier.h"
#include "UserContentControllerIdentifier.h"
#include "UserScriptIdentifier.h"
#include "UserStyleSheetIdentifier.h"
#include "WebScriptMessageHandler.h"
#include "WebUserContentControllerDataTypes.h"
#include <WebCore/UserContentProvider.h>
#include <wtf/HashMap.h>

#if ENABLE(CONTENT_EXTENSIONS)
#include <WebCore/ContentExtensionsBackend.h>
#endif

namespace WebCore {
namespace ContentExtensions {
class CompiledContentExtension;
}
}

namespace WebKit {

class InjectedBundleScriptWorld;
class WebCompiledContentRuleListData;
class WebUserMessageHandlerDescriptorProxy;
enum class InjectUserScriptImmediately : bool;

class WebUserContentController final : public WebCore::UserContentProvider, public IPC::MessageReceiver {
public:
    static Ref<WebUserContentController> getOrCreate(UserContentControllerIdentifier);
    virtual ~WebUserContentController();

    void ref() const final { WebCore::UserContentProvider::ref(); }
    void deref() const final { WebCore::UserContentProvider::deref(); }

    UserContentControllerIdentifier identifier() { return m_identifier; }

    void addUserScript(InjectedBundleScriptWorld&, WebCore::UserScript&&);
    void removeUserScriptWithURL(InjectedBundleScriptWorld&, const URL&);
    void removeUserScripts(InjectedBundleScriptWorld&);
    void addUserStyleSheet(InjectedBundleScriptWorld&, WebCore::UserStyleSheet&&);
    void removeUserStyleSheetWithURL(InjectedBundleScriptWorld&, const URL&);
    void removeUserStyleSheets(InjectedBundleScriptWorld&);
    void removeAllUserContent();

    InjectedBundleScriptWorld* worldForIdentifier(ContentWorldIdentifier);

    void addContentWorlds(const Vector<ContentWorldData>&);
    InjectedBundleScriptWorld* addContentWorld(const ContentWorldData&);
    void addUserScripts(Vector<WebUserScriptData>&&, InjectUserScriptImmediately);
    void addUserStyleSheets(const Vector<WebUserStyleSheetData>&);
    void addUserScriptMessageHandlers(const Vector<WebScriptMessageHandlerData>&);
#if ENABLE(CONTENT_EXTENSIONS)
    void addContentRuleLists(Vector<std::pair<WebCompiledContentRuleListData, URL>>&&);
#endif

private:
    explicit WebUserContentController(UserContentControllerIdentifier);

    // WebCore::UserContentProvider
    void forEachUserScript(Function<void(WebCore::DOMWrapperWorld&, const WebCore::UserScript&)>&&) const final;
    void forEachUserStyleSheet(Function<void(const WebCore::UserStyleSheet&)>&&) const final;
#if ENABLE(USER_MESSAGE_HANDLERS)
    void forEachUserMessageHandler(Function<void(const WebCore::UserMessageHandlerDescriptor&)>&&) const final;
#endif
#if ENABLE(CONTENT_EXTENSIONS)
    WebCore::ContentExtensions::ContentExtensionsBackend& userContentExtensionBackend() override { return m_contentExtensionBackend; }
#endif

    // IPC::MessageReceiver.
    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) override;

    void removeContentWorlds(const Vector<ContentWorldIdentifier>&);

    void removeUserScript(ContentWorldIdentifier, UserScriptIdentifier);
    void removeAllUserScripts(const Vector<ContentWorldIdentifier>&);

    void removeUserStyleSheet(ContentWorldIdentifier, UserStyleSheetIdentifier);
    void removeAllUserStyleSheets(const Vector<ContentWorldIdentifier>&);

    void removeUserScriptMessageHandler(ContentWorldIdentifier, ScriptMessageHandlerIdentifier);
    void removeAllUserScriptMessageHandlersForWorlds(const Vector<ContentWorldIdentifier>&);
    void removeAllUserScriptMessageHandlers();

#if ENABLE(CONTENT_EXTENSIONS)
    void removeContentRuleList(const String& name);
    void removeAllContentRuleLists();
#endif

    void addUserScriptInternal(InjectedBundleScriptWorld&, const std::optional<UserScriptIdentifier>&, WebCore::UserScript&&, InjectUserScriptImmediately);
    void removeUserScriptInternal(InjectedBundleScriptWorld&, UserScriptIdentifier);
    void addUserStyleSheetInternal(InjectedBundleScriptWorld&, const std::optional<UserStyleSheetIdentifier>&, WebCore::UserStyleSheet&&);
    void removeUserStyleSheetInternal(InjectedBundleScriptWorld&, UserStyleSheetIdentifier);
#if ENABLE(USER_MESSAGE_HANDLERS)
    void addUserScriptMessageHandlerInternal(InjectedBundleScriptWorld&, ScriptMessageHandlerIdentifier, const AtomString& name);
    void removeUserScriptMessageHandlerInternal(InjectedBundleScriptWorld&, ScriptMessageHandlerIdentifier);
#endif

    UserContentControllerIdentifier m_identifier;

    typedef HashMap<RefPtr<InjectedBundleScriptWorld>, Vector<std::pair<std::optional<UserScriptIdentifier>, WebCore::UserScript>>> WorldToUserScriptMap;
    WorldToUserScriptMap m_userScripts;

    typedef HashMap<RefPtr<InjectedBundleScriptWorld>, Vector<std::pair<std::optional<UserStyleSheetIdentifier>, WebCore::UserStyleSheet>>> WorldToUserStyleSheetMap;
    WorldToUserStyleSheetMap m_userStyleSheets;

#if ENABLE(USER_MESSAGE_HANDLERS)
    typedef HashMap<RefPtr<InjectedBundleScriptWorld>, Vector<std::pair<ScriptMessageHandlerIdentifier, RefPtr<WebUserMessageHandlerDescriptorProxy>>>> WorldToUserMessageHandlerVectorMap;
    WorldToUserMessageHandlerVectorMap m_userMessageHandlers;
#endif
#if ENABLE(CONTENT_EXTENSIONS)
    WebCore::ContentExtensions::ContentExtensionsBackend m_contentExtensionBackend;
#endif
};

} // namespace WebKit
