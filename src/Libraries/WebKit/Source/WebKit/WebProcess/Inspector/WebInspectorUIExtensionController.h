/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 30, 2025.
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

#include "Connection.h"
#include "InspectorExtensionTypes.h"
#include "MessageReceiver.h"
#include <WebCore/FrameIdentifier.h>
#include <WebCore/InspectorFrontendAPIDispatcher.h>
#include <WebCore/PageIdentifier.h>
#include <wtf/Forward.h>
#include <wtf/Noncopyable.h>
#include <wtf/RefCounted.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/URL.h>
#include <wtf/WeakPtr.h>

namespace JSC {
class JSValue;
class JSObject;
}

namespace WebCore {
class InspectorFrontendClient;
}

namespace WebKit {

class WebInspectorUI;

class WebInspectorUIExtensionController
    : public IPC::MessageReceiver
    , public RefCounted<WebInspectorUIExtensionController> {
    WTF_MAKE_TZONE_ALLOCATED(WebInspectorUIExtensionController);
    WTF_MAKE_NONCOPYABLE(WebInspectorUIExtensionController);
public:
    static Ref<WebInspectorUIExtensionController> create(WebCore::InspectorFrontendClient& inspectorFrontend, WebCore::PageIdentifier pageIdentifier)
    {
        return adoptRef(*new WebInspectorUIExtensionController(inspectorFrontend, pageIdentifier));
    }

    ~WebInspectorUIExtensionController();

    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    // Implemented in generated WebInspectorUIExtensionControllerMessageReceiver.cpp
    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) override;

    // WebInspectorUIExtensionController IPC messages.
    void registerExtension(const Inspector::ExtensionID&, const String& extensionBundleIdentifier, const String& displayName, CompletionHandler<void(Expected<void, Inspector::ExtensionError>)>&&);
    void unregisterExtension(const Inspector::ExtensionID&, CompletionHandler<void(Expected<void, Inspector::ExtensionError>)>&&);
    void createTabForExtension(const Inspector::ExtensionID&, const String& tabName, const URL& tabIconURL, const URL& sourceURL, CompletionHandler<void(Expected<Inspector::ExtensionTabID, Inspector::ExtensionError>)>&&);
    void evaluateScriptForExtension(const Inspector::ExtensionID&, const String& scriptSource, const std::optional<URL>& frameURL, const std::optional<URL>& contextSecurityOrigin, const std::optional<bool>& useContentScriptContext, CompletionHandler<void(std::span<const uint8_t>, const std::optional<WebCore::ExceptionDetails>&, const std::optional<Inspector::ExtensionError>&)>&&);
    void reloadForExtension(const Inspector::ExtensionID&, const std::optional<bool>& ignoreCache, const std::optional<String>& userAgent, const std::optional<String>& injectedScript, CompletionHandler<void(const std::optional<Inspector::ExtensionError>&)>&&);
    void showExtensionTab(const Inspector::ExtensionTabID&, CompletionHandler<void(Expected<void, Inspector::ExtensionError>)>&&);
    void navigateTabForExtension(const Inspector::ExtensionTabID&, const URL& sourceURL, CompletionHandler<void(const std::optional<Inspector::ExtensionError>&)>&&);

    // WebInspectorUIExtensionController IPC messages for testing.
    void evaluateScriptInExtensionTab(const Inspector::ExtensionTabID&, const String& scriptSource, CompletionHandler<void(std::span<const uint8_t>, const std::optional<WebCore::ExceptionDetails>&, const std::optional<Inspector::ExtensionError>&)>&&);

    // Callbacks from the frontend.
    void didShowExtensionTab(const Inspector::ExtensionID&, const Inspector::ExtensionTabID&, WebCore::FrameIdentifier);
    void didHideExtensionTab(const Inspector::ExtensionID&, const Inspector::ExtensionTabID&);
    void didNavigateExtensionTab(const Inspector::ExtensionID&, const Inspector::ExtensionTabID&, const URL&);
    void inspectedPageDidNavigate(const URL&);

private:
    WebInspectorUIExtensionController(WebCore::InspectorFrontendClient&, WebCore::PageIdentifier);

    JSC::JSObject* unwrapEvaluationResultAsObject(WebCore::InspectorFrontendAPIDispatcher::EvaluationResult) const;
    std::optional<Inspector::ExtensionError> parseExtensionErrorFromEvaluationResult(WebCore::InspectorFrontendAPIDispatcher::EvaluationResult) const;

    WeakPtr<WebCore::InspectorFrontendClient> m_frontendClient;
    WebCore::PageIdentifier m_inspectorPageIdentifier;
};

} // namespace WebKit

#endif // ENABLE(INSPECTOR_EXTENSIONS)
