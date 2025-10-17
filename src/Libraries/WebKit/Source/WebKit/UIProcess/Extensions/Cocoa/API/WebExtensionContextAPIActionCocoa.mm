/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 20, 2023.
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

#import "CocoaHelpers.h"
#import "WebExtensionAction.h"
#import "WebExtensionContextProxyMessages.h"
#import "WebExtensionUtilities.h"

namespace WebKit {

static Expected<Ref<WebExtensionAction>, WebExtensionError> getActionWithIdentifiers(std::optional<WebExtensionWindowIdentifier> windowIdentifier, std::optional<WebExtensionTabIdentifier> tabIdentifier, WebExtensionContext& context, NSString *apiName)
{
    if (windowIdentifier) {
        RefPtr window = context.getWindow(windowIdentifier.value());
        if (!window)
            return toWebExtensionError(apiName, nullString(), @"window not found");

        return context.getAction(window.get());
    }

    if (tabIdentifier) {
        RefPtr tab = context.getTab(tabIdentifier.value());
        if (!tab)
            return toWebExtensionError(apiName, nullString(), @"tab not found");

        return context.getAction(tab.get());
    }

    return Ref { context.defaultAction() };
}

static Expected<Ref<WebExtensionAction>, WebExtensionError> getOrCreateActionWithIdentifiers(std::optional<WebExtensionWindowIdentifier> windowIdentifier, std::optional<WebExtensionTabIdentifier> tabIdentifier, WebExtensionContext& context, NSString *apiName)
{
    if (windowIdentifier) {
        RefPtr window = context.getWindow(windowIdentifier.value());
        if (!window)
            return toWebExtensionError(apiName, nullString(), @"window not found");

        return context.getOrCreateAction(window.get());
    }

    if (tabIdentifier) {
        RefPtr tab = context.getTab(tabIdentifier.value());
        if (!tab)
            return toWebExtensionError(apiName, nullString(), @"tab not found");

        return context.getOrCreateAction(tab.get());
    }

    return Ref { context.defaultAction() };
}

bool WebExtensionContext::isActionMessageAllowed()
{
    Ref extension = *m_extension;
    return isLoaded() && (extension->hasAction() || extension->hasBrowserAction() || extension->hasPageAction());
}

void WebExtensionContext::actionGetTitle(std::optional<WebExtensionWindowIdentifier> windowIdentifier, std::optional<WebExtensionTabIdentifier> tabIdentifier, CompletionHandler<void(Expected<String, WebExtensionError>&&)>&& completionHandler)
{
    static NSString * const apiName = @"action.getTitle()";

    auto action = getActionWithIdentifiers(windowIdentifier, tabIdentifier, *this, apiName);
    if (!action) {
        completionHandler(makeUnexpected(action.error()));
        return;
    }

    completionHandler(Ref { action.value() }->label(WebExtensionAction::FallbackWhenEmpty::No));
}

void WebExtensionContext::actionSetTitle(std::optional<WebExtensionWindowIdentifier> windowIdentifier, std::optional<WebExtensionTabIdentifier> tabIdentifier, const String& title, CompletionHandler<void(Expected<void, WebExtensionError>&&)>&& completionHandler)
{
    static NSString * const apiName = @"action.setTitle()";

    auto action = getOrCreateActionWithIdentifiers(windowIdentifier, tabIdentifier, *this, apiName);
    if (!action) {
        completionHandler(makeUnexpected(action.error()));
        return;
    }

    Ref { action.value() }->setLabel(title);

    completionHandler({ });
}

void WebExtensionContext::actionSetIcon(std::optional<WebExtensionWindowIdentifier> windowIdentifier, std::optional<WebExtensionTabIdentifier> tabIdentifier, const String& iconsJSON, CompletionHandler<void(Expected<void, WebExtensionError>&&)>&& completionHandler)
{
    static NSString * const apiName = @"action.setIcon()";

    auto action = getOrCreateActionWithIdentifiers(windowIdentifier, tabIdentifier, *this, apiName);
    if (!action) {
        completionHandler(makeUnexpected(action.error()));
        return;
    }

    RefPtr parsedIcons = JSON::Value::parseJSON(iconsJSON);
    Ref webExtensionAction = action.value();

    if (RefPtr object = parsedIcons->asObject())
        webExtensionAction->setIcons(object);
#if ENABLE(WK_WEB_EXTENSIONS_ICON_VARIANTS)
    else if (RefPtr array = parsedIcons->asArray())
        webExtensionAction->setIconVariants(array);
#endif
    else {
        webExtensionAction->setIcons(nullptr);
#if ENABLE(WK_WEB_EXTENSIONS_ICON_VARIANTS)
        webExtensionAction->setIconVariants(nullptr);
#endif
    }

    completionHandler({ });
}

void WebExtensionContext::actionGetPopup(std::optional<WebExtensionWindowIdentifier> windowIdentifier, std::optional<WebExtensionTabIdentifier> tabIdentifier, CompletionHandler<void(Expected<String, WebExtensionError>&&)>&& completionHandler)
{
    static NSString * const apiName = @"action.getPopup()";

    auto action = getActionWithIdentifiers(windowIdentifier, tabIdentifier, *this, apiName);
    if (!action) {
        completionHandler(makeUnexpected(action.error()));
        return;
    }

    completionHandler(Ref { action.value() }->popupPath());
}

void WebExtensionContext::actionSetPopup(std::optional<WebExtensionWindowIdentifier> windowIdentifier, std::optional<WebExtensionTabIdentifier> tabIdentifier, const String& popupPath, CompletionHandler<void(Expected<void, WebExtensionError>&&)>&& completionHandler)
{
    static NSString * const apiName = @"action.setPopup()";

    auto action = getOrCreateActionWithIdentifiers(windowIdentifier, tabIdentifier, *this, apiName);
    if (!action) {
        completionHandler(makeUnexpected(action.error()));
        return;
    }

    Ref { action.value() }->setPopupPath(popupPath);

    completionHandler({ });
}

void WebExtensionContext::actionOpenPopup(WebPageProxyIdentifier identifier, std::optional<WebExtensionWindowIdentifier> windowIdentifier, std::optional<WebExtensionTabIdentifier> tabIdentifier, CompletionHandler<void(Expected<void, WebExtensionError>&&)>&& completionHandler)
{
    static NSString * const apiName = @"action.openPopup()";

    if (!protectedDefaultAction()->canProgrammaticallyPresentPopup()) {
        completionHandler(toWebExtensionError(apiName, nullString(), @"it is not implemented"));
        return;
    }

    if (extensionController()->isShowingActionPopup()) {
        completionHandler(toWebExtensionError(apiName, nullString(), @"another popup is already open"));
        return;
    }

    RefPtr<WebExtensionWindow> window;
    RefPtr<WebExtensionTab> tab;

    if (windowIdentifier) {
        window = getWindow(windowIdentifier.value());
        if (!window) {
            completionHandler(toWebExtensionError(apiName, nullString(), @"window not found"));
            return;
        }

        tab = window->activeTab();
        if (!tab) {
            completionHandler(toWebExtensionError(apiName, nullString(), @"active tab not found in window"));
            return;
        }
    }

    if (tabIdentifier) {
        tab = getTab(tabIdentifier.value());
        if (!tab) {
            completionHandler(toWebExtensionError(apiName, nullString(), @"tab not found"));
            return;
        }
    }

    if (!tab) {
        window = frontmostWindow();
        if (!window) {
            completionHandler(toWebExtensionError(apiName, nullString(), @"no windows open"));
            return;
        }

        tab = window->activeTab();
        if (!tab) {
            completionHandler(toWebExtensionError(apiName, nullString(), @"active tab not found in window"));
            return;
        }
    }

    if (getOrCreateAction(tab.get())->presentsPopup())
        performAction(tab.get(), UserTriggered::No);

    completionHandler({ });
}

void WebExtensionContext::actionGetBadgeText(std::optional<WebExtensionWindowIdentifier> windowIdentifier, std::optional<WebExtensionTabIdentifier> tabIdentifier, CompletionHandler<void(Expected<String, WebExtensionError>&&)>&& completionHandler)
{
    static NSString * const apiName = @"action.getBadgeText()";

    auto action = getActionWithIdentifiers(windowIdentifier, tabIdentifier, *this, apiName);
    if (!action) {
        completionHandler(makeUnexpected(action.error()));
        return;
    }

    completionHandler(Ref { action.value() }->badgeText());
}

void WebExtensionContext::actionSetBadgeText(std::optional<WebExtensionWindowIdentifier> windowIdentifier, std::optional<WebExtensionTabIdentifier> tabIdentifier, const String& text, CompletionHandler<void(Expected<void, WebExtensionError>&&)>&& completionHandler)
{
    static NSString * const apiName = @"action.setBadgeText()";

    auto action = getOrCreateActionWithIdentifiers(windowIdentifier, tabIdentifier, *this, apiName);
    if (!action) {
        completionHandler(makeUnexpected(action.error()));
        return;
    }

    Ref { action.value() }->setBadgeText(text);

    completionHandler({ });
}

void WebExtensionContext::actionGetEnabled(std::optional<WebExtensionWindowIdentifier> windowIdentifier, std::optional<WebExtensionTabIdentifier> tabIdentifier, CompletionHandler<void(Expected<bool, WebExtensionError>&&)>&& completionHandler)
{
    static NSString * const apiName = @"action.isEnabled()";

    auto action = getActionWithIdentifiers(windowIdentifier, tabIdentifier, *this, apiName);
    if (!action) {
        completionHandler(makeUnexpected(action.error()));
        return;
    }

    completionHandler(Ref { action.value() }->isEnabled());
}

void WebExtensionContext::actionSetEnabled(std::optional<WebExtensionTabIdentifier> tabIdentifier, bool enabled, CompletionHandler<void(Expected<void, WebExtensionError>&&)>&& completionHandler)
{
    auto action = getOrCreateActionWithIdentifiers(std::nullopt, tabIdentifier, *this, enabled ? @"action.enable()" : @"action.disable()");
    if (!action) {
        completionHandler(makeUnexpected(action.error()));
        return;
    }

    Ref { action.value() }->setEnabled(enabled);

    completionHandler({ });
}

void WebExtensionContext::fireActionClickedEventIfNeeded(WebExtensionTab* tab)
{
    constexpr auto type = WebExtensionEventListenerType::ActionOnClicked;
    wakeUpBackgroundContentIfNecessaryToFireEvents({ type }, [=, this, protectedThis = Ref { *this }, tab = RefPtr { tab }] {
        sendToProcessesForEvent(type, Messages::WebExtensionContextProxy::DispatchActionClickedEvent(tab ? std::optional(tab->parameters()) : std::nullopt));
    });
}

} // namespace WebKit

#endif // ENABLE(WK_WEB_EXTENSIONS)
