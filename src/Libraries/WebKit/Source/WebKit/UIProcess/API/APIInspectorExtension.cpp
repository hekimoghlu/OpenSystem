/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 12, 2024.
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
#include "config.h"
#include "APIInspectorExtension.h"

#if ENABLE(INSPECTOR_EXTENSIONS)

#include "APISerializedScriptValue.h"
#include "InspectorExtensionTypes.h"
#include "WebInspectorUIExtensionControllerProxy.h"
#include <WebCore/ExceptionDetails.h>
#include <wtf/UniqueRef.h>

namespace API {

InspectorExtension::InspectorExtension(const WTF::String& identifier, WebKit::WebInspectorUIExtensionControllerProxy& extensionControllerProxy)
    : m_identifier(identifier)
    , m_extensionControllerProxy(extensionControllerProxy)
{
}

Ref<InspectorExtension> InspectorExtension::create(const WTF::String& identifier, WebKit::WebInspectorUIExtensionControllerProxy& extensionControllerProxy)
{
    return adoptRef(*new InspectorExtension(identifier, extensionControllerProxy));
}

void InspectorExtension::setClient(UniqueRef<InspectorExtensionClient>&& client)
{
    m_client = client.moveToUniquePtr();
}

void InspectorExtension::createTab(const WTF::String& tabName, const WTF::URL& tabIconURL, const WTF::URL& sourceURL, WTF::CompletionHandler<void(Expected<Inspector::ExtensionTabID, Inspector::ExtensionError>)>&& completionHandler)
{
    if (!m_extensionControllerProxy) {
        completionHandler(makeUnexpected(Inspector::ExtensionError::ContextDestroyed));
        return;
    }

    m_extensionControllerProxy->createTabForExtension(m_identifier, tabName, tabIconURL, sourceURL, WTFMove(completionHandler));
}

void InspectorExtension::evaluateScript(const WTF::String& scriptSource, const std::optional<WTF::URL>& frameURL, const std::optional<WTF::URL>& contextSecurityOrigin, const std::optional<bool>& useContentScriptContext, WTF::CompletionHandler<void(Inspector::ExtensionEvaluationResult)>&& completionHandler)
{
    if (!m_extensionControllerProxy) {
        completionHandler(makeUnexpected(Inspector::ExtensionError::ContextDestroyed));
        return;
    }

    m_extensionControllerProxy->evaluateScriptForExtension(m_identifier, scriptSource, frameURL, contextSecurityOrigin, useContentScriptContext, WTFMove(completionHandler));
}

void InspectorExtension::navigateTab(const Inspector::ExtensionTabID& extensionTabID, const WTF::URL& sourceURL, WTF::CompletionHandler<void(const std::optional<Inspector::ExtensionError>)>&& completionHandler)
{
    if (!m_extensionControllerProxy) {
        completionHandler(Inspector::ExtensionError::ContextDestroyed);
        return;
    }

    m_extensionControllerProxy->navigateTabForExtension(extensionTabID, sourceURL, WTFMove(completionHandler));
}

void InspectorExtension::reloadIgnoringCache(const std::optional<bool>& ignoreCache, const std::optional<WTF::String>& userAgent, const std::optional<WTF::String>& injectedScript,  WTF::CompletionHandler<void(Inspector::ExtensionVoidResult)>&& completionHandler)
{
    if (!m_extensionControllerProxy) {
        completionHandler(makeUnexpected(Inspector::ExtensionError::ContextDestroyed));
        return;
    }

    m_extensionControllerProxy->reloadForExtension(m_identifier, ignoreCache, userAgent, injectedScript, WTFMove(completionHandler));
}

// For testing.

void InspectorExtension::evaluateScriptInExtensionTab(const Inspector::ExtensionTabID& extensionTabID, const WTF::String& scriptSource, WTF::CompletionHandler<void(Inspector::ExtensionEvaluationResult)>&& completionHandler)
{
    if (!m_extensionControllerProxy) {
        completionHandler(makeUnexpected(Inspector::ExtensionError::ContextDestroyed));
        return;
    }

    m_extensionControllerProxy->evaluateScriptInExtensionTab(extensionTabID, scriptSource, WTFMove(completionHandler));
}

} // namespace API

#endif // ENABLE(INSPECTOR_EXTENSIONS)
