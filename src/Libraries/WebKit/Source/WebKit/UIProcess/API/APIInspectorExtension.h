/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 12, 2025.
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

#include "APIInspectorExtensionClient.h"
#include "APIObject.h"
#include "InspectorExtensionTypes.h"
#include <wtf/CompletionHandler.h>
#include <wtf/Forward.h>
#include <wtf/WeakPtr.h>
#include <wtf/text/WTFString.h>

namespace WebKit {
class WebInspectorUIExtensionControllerProxy;
}

namespace API {

class InspectorExtension final : public API::ObjectImpl<Object::Type::InspectorExtension> {
public:
    static Ref<InspectorExtension> create(const WTF::String& identifier, WebKit::WebInspectorUIExtensionControllerProxy&);

    const WTF::String& identifier() const { return m_identifier; }

    void createTab(const WTF::String& tabName, const WTF::URL& tabIconURL, const WTF::URL& sourceURL, WTF::CompletionHandler<void(Expected<Inspector::ExtensionTabID, Inspector::ExtensionError>)>&&);
    void evaluateScript(const WTF::String& scriptSource, const std::optional<WTF::URL>& frameURL, const std::optional<WTF::URL>& contextSecurityOrigin, const std::optional<bool>& useContentScriptContext, WTF::CompletionHandler<void(Inspector::ExtensionEvaluationResult)>&&);
    void navigateTab(const Inspector::ExtensionTabID&, const WTF::URL& sourceURL, WTF::CompletionHandler<void(const std::optional<Inspector::ExtensionError>)>&&);
    void reloadIgnoringCache(const std::optional<bool>& ignoreCache, const std::optional<WTF::String>& userAgent, const std::optional<WTF::String>& injectedScript, WTF::CompletionHandler<void(Inspector::ExtensionVoidResult)>&&);

    // For testing.
    void evaluateScriptInExtensionTab(const Inspector::ExtensionTabID&, const WTF::String& scriptSource, WTF::CompletionHandler<void(Inspector::ExtensionEvaluationResult)>&&);

    InspectorExtensionClient* client() const { return m_client.get(); }
    void setClient(UniqueRef<InspectorExtensionClient>&&);

private:
    InspectorExtension(const WTF::String& identifier, WebKit::WebInspectorUIExtensionControllerProxy&);

    WTF::String m_identifier;
    WeakPtr<WebKit::WebInspectorUIExtensionControllerProxy> m_extensionControllerProxy;
    std::unique_ptr<API::InspectorExtensionClient> m_client;
};

} // namespace API

#endif // ENABLE(INSPECTOR_EXTENSIONS)
