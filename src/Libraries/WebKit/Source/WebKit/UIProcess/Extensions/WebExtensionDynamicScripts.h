/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 27, 2021.
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

#if ENABLE(WK_WEB_EXTENSIONS)

#include "APIContentWorld.h"
#include "APIUserScript.h"
#include "APIUserStyleSheet.h"
#include "WebExtension.h"
#include "WebExtensionFrameIdentifier.h"
#include "WebExtensionRegisteredScriptParameters.h"
#include "WebExtensionScriptInjectionResultParameters.h"
#include <wtf/Forward.h>
#include <wtf/RefCounted.h>
#include <wtf/TZoneMalloc.h>

OBJC_CLASS WKWebView;
OBJC_CLASS WKFrameInfo;
OBJC_CLASS _WKFrameTreeNode;

namespace WebKit {

class WebExtensionContext;
class WebExtensionTab;
struct WebExtensionScriptInjectionParameters;

namespace WebExtensionDynamicScripts {

using InjectionResults = Vector<WebExtensionScriptInjectionResultParameters>;

using SourcePair = std::pair<String, URL>;
using SourcePairs = Vector<SourcePair>;

using InjectionTime = WebExtension::InjectionTime;
using InjectedContentData = WebExtension::InjectedContentData;

using UserScriptVector = Vector<Ref<API::UserScript>>;
using UserStyleSheetVector = Vector<Ref<API::UserStyleSheet>>;

class WebExtensionRegisteredScript : public RefCounted<WebExtensionRegisteredScript> {
    WTF_MAKE_NONCOPYABLE(WebExtensionRegisteredScript);
    WTF_MAKE_TZONE_ALLOCATED(WebExtensionRegisteredScript);

public:
    template<typename... Args>
    static Ref<WebExtensionRegisteredScript> create(Args&&... args)
    {
        return adoptRef(*new WebExtensionRegisteredScript(std::forward<Args>(args)...));
    }

    enum class FirstTimeRegistration { No, Yes };

    void updateParameters(const WebExtensionRegisteredScriptParameters&);

    void merge(WebExtensionRegisteredScriptParameters&);

    void addUserScript(const String& identifier, API::UserScript&);
    void addUserStyleSheet(const String& identifier, API::UserStyleSheet&);
    void removeUserScriptsAndStyleSheets(const String& identifier);

    void updateInjectedContent(InjectedContentData& injectedContent) { m_injectedContent = injectedContent; }
    const InjectedContentData& injectedContent() const { return m_injectedContent; }

    WebExtensionRegisteredScriptParameters parameters() const { return m_parameters; };

private:
    explicit WebExtensionRegisteredScript(WebExtensionContext& extensionContext, const WebExtensionRegisteredScriptParameters& parameters, const InjectedContentData& injectedContent)
        : m_extensionContext(extensionContext)
        , m_parameters(parameters)
        , m_injectedContent(injectedContent)
    {
    }

    WeakPtr<WebExtensionContext> m_extensionContext;
    WebExtensionRegisteredScriptParameters m_parameters;
    InjectedContentData m_injectedContent;

    HashMap<String, UserScriptVector> m_userScriptsMap;
    HashMap<String, UserStyleSheetVector> m_userStyleSheetsMap;

    void removeUserStyleSheets(const String& identifier);
    void removeUserScripts(const String& identifier);
};

std::optional<SourcePair> sourcePairForResource(const String& path, WebExtensionContext&);
SourcePairs getSourcePairsForParameters(const WebExtensionScriptInjectionParameters&, WebExtensionContext&);

void executeScript(const SourcePairs&, WKWebView *, API::ContentWorld&, WebExtensionTab&, const WebExtensionScriptInjectionParameters&, WebExtensionContext&, CompletionHandler<void(InjectionResults&&)>&&);
void injectStyleSheets(const SourcePairs&, WKWebView *, API::ContentWorld&, WebCore::UserStyleLevel, WebCore::UserContentInjectedFrames, WebExtensionContext&);
void removeStyleSheets(const SourcePairs&, WKWebView *, WebCore::UserContentInjectedFrames, WebExtensionContext&);

#if PLATFORM(COCOA)
WebExtensionScriptInjectionResultParameters toInjectionResultParameters(id resultOfExecution, WKFrameInfo *, NSString *errorMessage);
#endif

} // namespace WebExtensionDynamicScripts

} // namespace WebKit

#endif // ENABLE(WK_WEB_EXTENSIONS)
