/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 8, 2022.
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

#if ENABLE(INSPECTOR_EXTENSIONS)

#include "InspectorExtensionTypes.h"
#include "MessageReceiver.h"
#include <WebCore/FrameIdentifier.h>
#include <wtf/Expected.h>
#include <wtf/Forward.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/URL.h>
#include <wtf/WeakPtr.h>

namespace API {
class InspectorExtension;
}

namespace WebKit {

class WebPageProxy;

class WebInspectorUIExtensionControllerProxy final
    : public RefCounted<WebInspectorUIExtensionControllerProxy>
    , public IPC::MessageReceiver {
    WTF_MAKE_TZONE_ALLOCATED(WebInspectorUIExtensionControllerProxy);
    WTF_MAKE_NONCOPYABLE(WebInspectorUIExtensionControllerProxy);
public:
    static Ref<WebInspectorUIExtensionControllerProxy> create(WebPageProxy& inspectorPage);
    virtual ~WebInspectorUIExtensionControllerProxy();

    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    // Implemented in generated WebInspectorUIExtensionControllerProxyMessageReceiver.cpp
    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) override;

    // API.
    void registerExtension(const Inspector::ExtensionID&, const String& extensionBundleIdentifier, const String& displayName, WTF::CompletionHandler<void(Expected<RefPtr<API::InspectorExtension>, Inspector::ExtensionError>)>&&);
    void unregisterExtension(const Inspector::ExtensionID&, WTF::CompletionHandler<void(Expected<void, Inspector::ExtensionError>)>&&);
    void createTabForExtension(const Inspector::ExtensionID&, const String& tabName, const URL& tabIconURL, const URL& sourceURL, WTF::CompletionHandler<void(Expected<Inspector::ExtensionTabID, Inspector::ExtensionError>)>&&);
    void evaluateScriptForExtension(const Inspector::ExtensionID&, const String& scriptSource, const std::optional<URL>& frameURL, const std::optional<URL>& contextSecurityOrigin, const std::optional<bool>& useContentScriptContext, WTF::CompletionHandler<void(Inspector::ExtensionEvaluationResult)>&&);
    void reloadForExtension(const Inspector::ExtensionID&, const std::optional<bool>& ignoreCache, const std::optional<String>& userAgent, const std::optional<String>& injectedScript, WTF::CompletionHandler<void(Inspector::ExtensionVoidResult)>&&);
    void showExtensionTab(const Inspector::ExtensionTabID&, CompletionHandler<void(Expected<void, Inspector::ExtensionError>)>&&);
    void navigateTabForExtension(const Inspector::ExtensionTabID&, const URL& sourceURL, CompletionHandler<void(const std::optional<Inspector::ExtensionError>)>&&);
    // API for testing.
    void evaluateScriptInExtensionTab(const Inspector::ExtensionTabID&, const String& scriptSource, WTF::CompletionHandler<void(Inspector::ExtensionEvaluationResult)>&&);

    // WebInspectorUIExtensionControllerProxy IPC messages.
    void didShowExtensionTab(const Inspector::ExtensionID&, const Inspector::ExtensionTabID&, WebCore::FrameIdentifier);
    void didHideExtensionTab(const Inspector::ExtensionID&, const Inspector::ExtensionTabID&);
    void didNavigateExtensionTab(const Inspector::ExtensionID&, const Inspector::ExtensionTabID&, const URL&);
    void inspectedPageDidNavigate(const URL&);

    // Notifications.
    void inspectorFrontendLoaded();
    void inspectorFrontendWillClose();
    void effectiveAppearanceDidChange(Inspector::ExtensionAppearance);

private:
    explicit WebInspectorUIExtensionControllerProxy(WebPageProxy& inspectorPage);

    RefPtr<WebPageProxy> protectedInspectorPage() const;

    void whenFrontendHasLoaded(Function<void()>&&);

    WeakPtr<WebPageProxy> m_inspectorPage;
    HashMap<Inspector::ExtensionID, RefPtr<API::InspectorExtension>> m_extensionAPIObjectMap;

    // Used to queue actions such as registering extensions that happen early on.
    // There's no point sending these before the frontend is fully loaded.
    Vector<Function<void()>> m_frontendLoadedCallbackQueue;

    bool m_frontendLoaded { false };
};

} // namespace WebKit

#endif // ENABLE(INSPECTOR_EXTENSIONS)
