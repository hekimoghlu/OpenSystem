/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 13, 2022.
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
#if !__has_feature(objc_arc)
#error This file requires ARC. Add the "-fobjc-arc" compiler flag for this file.
#endif

#import "config.h"
#import "WebExtensionContextProxy.h"

#if ENABLE(WK_WEB_EXTENSIONS)

#import "CocoaHelpers.h"
#import "JSWebExtensionWrapper.h"
#import "WebExtensionAPINamespace.h"
#import "WebExtensionContextMessages.h"
#import "WebExtensionContextProxyMessages.h"
#import "WebExtensionControllerProxy.h"
#import "WebExtensionLocalization.h"
#import "WebPage.h"
#import "WebProcess.h"
#import <wtf/HashMap.h>
#import <wtf/NeverDestroyed.h>
#import <wtf/ObjectIdentifier.h>
#import <wtf/text/MakeString.h>

namespace WebKit {

static HashMap<WebExtensionContextIdentifier, WeakPtr<WebExtensionContextProxy>>& webExtensionContextProxies()
{
    static MainThreadNeverDestroyed<HashMap<WebExtensionContextIdentifier, WeakPtr<WebExtensionContextProxy>>> contexts;
    return contexts;
}

RefPtr<WebExtensionContextProxy> WebExtensionContextProxy::get(WebExtensionContextIdentifier identifier)
{
    return webExtensionContextProxies().get(identifier).get();
}

WebExtensionContextProxy::WebExtensionContextProxy(const WebExtensionContextParameters& parameters)
    : m_identifier(parameters.identifier)
{
    ASSERT(!get(m_identifier));
    webExtensionContextProxies().add(m_identifier, *this);

    WebProcess::singleton().addMessageReceiver(Messages::WebExtensionContextProxy::messageReceiverName(), m_identifier, *this);
}

WebExtensionContextProxy::~WebExtensionContextProxy()
{
    webExtensionContextProxies().remove(m_identifier);
    WebProcess::singleton().removeMessageReceiver(*this);
}

Ref<WebExtensionContextProxy> WebExtensionContextProxy::getOrCreate(const WebExtensionContextParameters& parameters, WebExtensionControllerProxy& extensionControllerProxy, WebPage* newPage)
{
    auto updateProperties = [&](WebExtensionContextProxy& context) {
        context.m_extensionControllerProxy = extensionControllerProxy;
        context.m_baseURL = parameters.baseURL;
        context.m_uniqueIdentifier = parameters.uniqueIdentifier;
        context.m_unsupportedAPIs = parameters.unsupportedAPIs;
        context.m_grantedPermissions = parameters.grantedPermissions;
        context.m_localization = parseLocalization(parameters.localizationJSON, parameters.baseURL);
        Ref manifestJSON = *parameters.manifestJSON;
        context.m_manifest = parseJSON(manifestJSON);
        context.m_manifestVersion = parameters.manifestVersion;
        context.m_isSessionStorageAllowedInContentScripts = parameters.isSessionStorageAllowedInContentScripts;

        if (parameters.backgroundPageIdentifier) {
            if (newPage && parameters.backgroundPageIdentifier.value() == newPage->identifier())
                context.setBackgroundPage(*newPage);
            else if (RefPtr page = WebProcess::singleton().webPage(parameters.backgroundPageIdentifier.value()))
                context.setBackgroundPage(*page);
        }

        auto processPageIdentifiers = [&context, &newPage](auto& identifiers, auto addPage) {
            for (auto& identifierTuple : identifiers) {
                auto& pageIdentifier = std::get<WebCore::PageIdentifier>(identifierTuple);
                auto& tabIdentifier = std::get<std::optional<WebExtensionTabIdentifier>>(identifierTuple);
                auto& windowIdentifier = std::get<std::optional<WebExtensionWindowIdentifier>>(identifierTuple);

                if (newPage && pageIdentifier == newPage->identifier())
                    addPage(context, *newPage, tabIdentifier, windowIdentifier);
                else if (RefPtr<WebPage> page = WebProcess::singleton().webPage(pageIdentifier))
                    addPage(context, *page, tabIdentifier, windowIdentifier);
            }
        };

#if ENABLE(INSPECTOR_EXTENSIONS)
        processPageIdentifiers(parameters.inspectorBackgroundPageIdentifiers, [](auto& context, auto& page, auto& tabIdentifier, auto& windowIdentifier) {
            context.addInspectorBackgroundPage(page, tabIdentifier, windowIdentifier);
        });
#endif

        processPageIdentifiers(parameters.popupPageIdentifiers, [](auto& context, auto& page, auto& tabIdentifier, auto& windowIdentifier) {
            context.addPopupPage(page, tabIdentifier, windowIdentifier);
        });

        processPageIdentifiers(parameters.tabPageIdentifiers, [](auto& context, auto& page, auto& tabIdentifier, auto& windowIdentifier) {
            context.addTabPage(page, tabIdentifier, windowIdentifier);
        });
    };

    if (RefPtr context = get(parameters.identifier)) {
        updateProperties(*context);
        return *context;
    }

    Ref result = adoptRef(*new WebExtensionContextProxy(parameters));
    updateProperties(result);
    return result;
}

bool WebExtensionContextProxy::isUnsupportedAPI(const String& propertyPath, const ASCIILiteral& propertyName) const
{
    auto fullPropertyPath = !propertyPath.isEmpty() ? makeString(propertyPath, '.', propertyName) : propertyName;
    return m_unsupportedAPIs.contains(fullPropertyPath);
}

bool WebExtensionContextProxy::hasPermission(const String& permission) const
{
    WallTime currentTime = WallTime::now();

    // If the next expiration date hasn't passed yet, there is nothing to remove.
    if (m_nextGrantedPermissionsExpirationDate != WallTime::nan() && m_nextGrantedPermissionsExpirationDate > currentTime)
        goto finish;

    m_nextGrantedPermissionsExpirationDate = WallTime::infinity();

    m_grantedPermissions.removeIf([&](auto& entry) {
        if (entry.value <= currentTime)
            return true;

        if (entry.value < m_nextGrantedPermissionsExpirationDate)
            m_nextGrantedPermissionsExpirationDate = entry.value;

        return false;
    });

finish:
    return m_grantedPermissions.contains(permission);
}

void WebExtensionContextProxy::updateGrantedPermissions(PermissionsMap&& permissions)
{
    m_grantedPermissions = WTFMove(permissions);
    m_nextGrantedPermissionsExpirationDate = WallTime::nan();
}

RefPtr<WebExtensionLocalization> WebExtensionContextProxy::parseLocalization(RefPtr<API::Data> json, const URL& baseURL)
{
    if (!json)
        return nullptr;

    if (RefPtr value = JSON::Value::parseJSON(String::fromUTF8(json->span()))) {
        if (RefPtr object = value->asObject())
            return WebExtensionLocalization::create(object, baseURL.host().toString());
    }

    return nullptr;
}

Ref<WebCore::DOMWrapperWorld> WebExtensionContextProxy::toDOMWrapperWorld(WebExtensionContentWorldType contentWorldType) const
{
    switch (contentWorldType) {
    case WebExtensionContentWorldType::Main:
    case WebExtensionContentWorldType::WebPage:
#if ENABLE(INSPECTOR_EXTENSIONS)
    case WebExtensionContentWorldType::Inspector:
#endif
        return mainWorldSingleton();
    case WebExtensionContentWorldType::ContentScript:
        return contentScriptWorld();
    case WebExtensionContentWorldType::Native:
        ASSERT_NOT_REACHED();
        return mainWorldSingleton();
    }
}

} // namespace WebKit

#endif // ENABLE(WK_WEB_EXTENSIONS)
