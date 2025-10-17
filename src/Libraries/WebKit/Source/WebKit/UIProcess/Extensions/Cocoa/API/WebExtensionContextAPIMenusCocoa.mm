/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 27, 2024.
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
#import "WebExtensionContext.h"

#if ENABLE(WK_WEB_EXTENSIONS)

#import "WebExtensionContextProxyMessages.h"
#import "WebExtensionMenuItem.h"
#import "WebExtensionMenuItemContextParameters.h"
#import "WebExtensionMenuItemParameters.h"
#import "WebExtensionUtilities.h"

namespace WebKit {

static bool isAncestorOrSelf(WebExtensionContext& context, const String& potentialAncestorIdentifier, const String& identifier)
{
    if (potentialAncestorIdentifier == identifier)
        return true;

    RefPtr current = context.menuItem(identifier);
    while (current) {
        RefPtr parent = current->parentMenuItem();
        if (parent && parent->identifier() == potentialAncestorIdentifier)
            return true;
        current = parent;
    }

    return false;
}

bool WebExtensionContext::isMenusMessageAllowed()
{
    return isLoaded() && (hasPermission(WKWebExtensionPermissionContextMenus) || hasPermission(WKWebExtensionPermissionMenus));
}

void WebExtensionContext::menusCreate(const WebExtensionMenuItemParameters& parameters, CompletionHandler<void(Expected<void, WebExtensionError>&&)>&& completionHandler)
{
    static NSString * const apiName = @"menus.create()";

    if (m_menuItems.contains(parameters.identifier)) {
        completionHandler(toWebExtensionError(apiName, nullString(), @"identifier is already used"));
        return;
    }

    if (parameters.parentIdentifier && !m_menuItems.contains(parameters.parentIdentifier.value())) {
        completionHandler(toWebExtensionError(apiName, nullString(), @"parent menu item not found"));
        return;
    }

    if (parameters.parentIdentifier && isAncestorOrSelf(*this, parameters.parentIdentifier.value(), parameters.identifier)) {
        completionHandler(toWebExtensionError(apiName, nullString(), @"parent menu item cannot be another ancestor"));
        return;
    }

    auto menuItem = WebExtensionMenuItem::create(*this, parameters);

    m_menuItems.set(parameters.identifier, menuItem);

    if (!parameters.parentIdentifier)
        m_mainMenuItems.append(menuItem);

    completionHandler({ });
}

void WebExtensionContext::menusUpdate(const String& identifier, const WebExtensionMenuItemParameters& parameters, CompletionHandler<void(Expected<void, WebExtensionError>&&)>&& completionHandler)
{
    static NSString * const apiName = @"menus.update()";

    RefPtr menuItem = this->menuItem(identifier);
    if (!menuItem) {
        completionHandler(toWebExtensionError(apiName, nullString(), @"menu item not found"));
        return;
    }

    if (!parameters.identifier.isEmpty() && identifier != parameters.identifier) {
        m_menuItems.remove(identifier);
        m_menuItems.set(parameters.identifier, *menuItem);
    }

    if (parameters.parentIdentifier && !m_menuItems.contains(parameters.parentIdentifier.value())) {
        completionHandler(toWebExtensionError(apiName, nullString(), @"parent menu item not found"));
        return;
    }

    if (parameters.parentIdentifier && isAncestorOrSelf(*this, parameters.parentIdentifier.value(), !parameters.identifier.isEmpty() ? parameters.identifier : identifier)) {
        completionHandler(toWebExtensionError(apiName, nullString(), @"parent menu item cannot be itself or another ancestor"));
        return;
    }

    menuItem->update(parameters);

    completionHandler({ });
}

void WebExtensionContext::menusRemove(const String& identifier, CompletionHandler<void(Expected<void, WebExtensionError>&&)>&& completionHandler)
{
    RefPtr menuItem = this->menuItem(identifier);
    if (!menuItem) {
        completionHandler(toWebExtensionError(@"menus.remove()", nullString(), @"menu item not found"));
        return;
    }

    Function<void(WebExtensionMenuItem&)> removeRecursive;
    removeRecursive = [this, protectedThis = Ref { *this }, &removeRecursive](WebExtensionMenuItem& menuItem) {
        for (auto& submenuItem : menuItem.submenuItems())
            removeRecursive(submenuItem);

        m_menuItems.remove(menuItem.identifier());

        if (!menuItem.parentMenuItem())
            m_mainMenuItems.removeAll(menuItem);
    };

    removeRecursive(*menuItem);

    completionHandler({ });
}

void WebExtensionContext::menusRemoveAll(CompletionHandler<void(Expected<void, WebExtensionError>&&)>&& completionHandler)
{
    m_menuItems.clear();
    m_mainMenuItems.clear();

    completionHandler({ });
}

void WebExtensionContext::fireMenusClickedEventIfNeeded(const WebExtensionMenuItem& menuItem, bool wasChecked, const WebExtensionMenuItemContextParameters& contextParameters)
{
    RefPtr tab = contextParameters.tabIdentifier ? getTab(contextParameters.tabIdentifier.value()) : nullptr;

    constexpr auto type = WebExtensionEventListenerType::MenusOnClicked;
    wakeUpBackgroundContentIfNecessaryToFireEvents({ type }, [=, this, protectedThis = Ref { *this }, menuItem = Ref { menuItem }] {
        sendToProcessesForEvent(type, Messages::WebExtensionContextProxy::DispatchMenusClickedEvent(menuItem->minimalParameters(), wasChecked, contextParameters, tab ? std::optional { tab->parameters() } : std::nullopt));
    });
}

} // namespace WebKit

#endif // ENABLE(WK_WEB_EXTENSIONS)
