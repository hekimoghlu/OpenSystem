/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 12, 2022.
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

#include "WebExtensionContentWorldType.h"

namespace WebKit {

// If you are adding a new event, you will also need to increase 'currentBackgroundContentListenerStateVersion'
// so that your new event gets fired to non-persistent background content.
enum class WebExtensionEventListenerType : uint8_t {
    Unknown = 0,
    ActionOnClicked,
    AlarmsOnAlarm,
    CommandsOnChanged,
    CommandsOnCommand,
    CookiesOnChanged,
    DevToolsElementsPanelOnSelectionChanged,
    DevToolsExtensionPanelOnHidden,
    DevToolsExtensionPanelOnSearch,
    DevToolsExtensionPanelOnShown,
    DevToolsExtensionSidebarPaneOnHidden,
    DevToolsExtensionSidebarPaneOnShown,
    DevToolsInspectedWindowOnResourceAdded,
    DevToolsNetworkOnNavigated,
    DevToolsNetworkOnRequestFinished,
    DevToolsPanelsOnThemeChanged,
    DownloadsOnChanged,
    DownloadsOnCreated,
    MenusOnClicked,
    NotificationsOnButtonClicked,
    NotificationsOnClicked,
    PermissionsOnAdded,
    PermissionsOnRemoved,
    PortOnDisconnect,
    PortOnMessage,
    RuntimeOnConnect,
    RuntimeOnConnectExternal,
    RuntimeOnInstalled,
    RuntimeOnMessage,
    RuntimeOnMessageExternal,
    RuntimeOnStartup,
    StorageOnChanged,
    TabsOnActivated,
    TabsOnAttached,
    TabsOnCreated,
    TabsOnDetached,
    TabsOnHighlighted,
    TabsOnMoved,
    TabsOnRemoved,
    TabsOnReplaced,
    TabsOnUpdated,
    TestOnMessage,
    WebNavigationOnBeforeNavigate,
    WebNavigationOnCommitted,
    WebNavigationOnCompleted,
    WebNavigationOnDOMContentLoaded,
    WebNavigationOnErrorOccurred,
    WebRequestOnAuthRequired,
    WebRequestOnBeforeRedirect,
    WebRequestOnBeforeRequest,
    WebRequestOnBeforeSendHeaders,
    WebRequestOnCompleted,
    WebRequestOnErrorOccurred,
    WebRequestOnHeadersReceived,
    WebRequestOnResponseStarted,
    WebRequestOnSendHeaders,
    WindowsOnCreated,
    WindowsOnFocusChanged,
    WindowsOnRemoved,
};

inline String toAPIString(WebExtensionEventListenerType eventType)
{
    switch (eventType) {
    case WebExtensionEventListenerType::Unknown:
        return nullString();
    case WebExtensionEventListenerType::ActionOnClicked:
        return "onClicked"_s;
    case WebExtensionEventListenerType::AlarmsOnAlarm:
        return "onAlarm"_s;
    case WebExtensionEventListenerType::CommandsOnChanged:
        return "onChanged"_s;
    case WebExtensionEventListenerType::CommandsOnCommand:
        return "onCommand"_s;
    case WebExtensionEventListenerType::CookiesOnChanged:
        return "onChanged"_s;
    case WebExtensionEventListenerType::DevToolsElementsPanelOnSelectionChanged:
        return "onSelectionChanged"_s;
    case WebExtensionEventListenerType::DevToolsExtensionPanelOnHidden:
        return "onHidden"_s;
    case WebExtensionEventListenerType::DevToolsExtensionPanelOnSearch:
        return "onSearch"_s;
    case WebExtensionEventListenerType::DevToolsExtensionPanelOnShown:
        return "onShown"_s;
    case WebExtensionEventListenerType::DevToolsExtensionSidebarPaneOnHidden:
        return "onHidden"_s;
    case WebExtensionEventListenerType::DevToolsExtensionSidebarPaneOnShown:
        return "onShown"_s;
    case WebExtensionEventListenerType::DevToolsInspectedWindowOnResourceAdded:
        return "onResourceAdded"_s;
    case WebExtensionEventListenerType::DevToolsNetworkOnNavigated:
        return "onNavigated"_s;
    case WebExtensionEventListenerType::DevToolsNetworkOnRequestFinished:
        return "onRequestFinished"_s;
    case WebExtensionEventListenerType::DevToolsPanelsOnThemeChanged:
        return "onThemeChanged"_s;
    case WebExtensionEventListenerType::DownloadsOnChanged:
        return "onChanged"_s;
    case WebExtensionEventListenerType::DownloadsOnCreated:
        return "onCreated"_s;
    case WebExtensionEventListenerType::MenusOnClicked:
        return "onClicked"_s;
    case WebExtensionEventListenerType::NotificationsOnButtonClicked:
        return "onButtonClicked"_s;
    case WebExtensionEventListenerType::NotificationsOnClicked:
        return "onClicked"_s;
    case WebExtensionEventListenerType::PermissionsOnAdded:
        return "onAdded"_s;
    case WebExtensionEventListenerType::PermissionsOnRemoved:
        return "onRemoved"_s;
    case WebExtensionEventListenerType::PortOnDisconnect:
        return "onDisconnect"_s;
    case WebExtensionEventListenerType::PortOnMessage:
        return "onMessage"_s;
    case WebExtensionEventListenerType::RuntimeOnConnect:
        return "onConnect"_s;
    case WebExtensionEventListenerType::RuntimeOnConnectExternal:
        return "onConnectExternal"_s;
    case WebExtensionEventListenerType::RuntimeOnInstalled:
        return "onInstalled"_s;
    case WebExtensionEventListenerType::RuntimeOnMessage:
        return "onMessage"_s;
    case WebExtensionEventListenerType::RuntimeOnMessageExternal:
        return "onMessageExternal"_s;
    case WebExtensionEventListenerType::RuntimeOnStartup:
        return "onStartup"_s;
    case WebExtensionEventListenerType::StorageOnChanged:
        return "onChanged"_s;
    case WebExtensionEventListenerType::TabsOnActivated:
        return "onActivated"_s;
    case WebExtensionEventListenerType::TabsOnAttached:
        return "onAttached"_s;
    case WebExtensionEventListenerType::TabsOnCreated:
        return "onCreated"_s;
    case WebExtensionEventListenerType::TabsOnDetached:
        return "onDetached"_s;
    case WebExtensionEventListenerType::TabsOnHighlighted:
        return "onHighlighted"_s;
    case WebExtensionEventListenerType::TabsOnMoved:
        return "onMoved"_s;
    case WebExtensionEventListenerType::TabsOnRemoved:
        return "onRemoved"_s;
    case WebExtensionEventListenerType::TabsOnReplaced:
        return "onReplaced"_s;
    case WebExtensionEventListenerType::TabsOnUpdated:
        return "onUpdated"_s;
    case WebExtensionEventListenerType::TestOnMessage:
        return "onMessage"_s;
    case WebExtensionEventListenerType::WebNavigationOnBeforeNavigate:
        return "onBeforeNavigate"_s;
    case WebExtensionEventListenerType::WebNavigationOnCommitted:
        return "onCommitted"_s;
    case WebExtensionEventListenerType::WebNavigationOnCompleted:
        return "onCompleted"_s;
    case WebExtensionEventListenerType::WebNavigationOnDOMContentLoaded:
        return "onDOMContentLoaded"_s;
    case WebExtensionEventListenerType::WebNavigationOnErrorOccurred:
        return "onErrorOccurred"_s;
    case WebExtensionEventListenerType::WebRequestOnAuthRequired:
        return "onAuthRequired"_s;
    case WebExtensionEventListenerType::WebRequestOnBeforeRedirect:
        return "onBeforeRedirect"_s;
    case WebExtensionEventListenerType::WebRequestOnBeforeRequest:
        return "onBeforeRequest"_s;
    case WebExtensionEventListenerType::WebRequestOnBeforeSendHeaders:
        return "onBeforeSendHeaders"_s;
    case WebExtensionEventListenerType::WebRequestOnCompleted:
        return "onCompleted"_s;
    case WebExtensionEventListenerType::WebRequestOnErrorOccurred:
        return "onErrorOccurred"_s;
    case WebExtensionEventListenerType::WebRequestOnHeadersReceived:
        return "onHeadersReceived"_s;
    case WebExtensionEventListenerType::WebRequestOnResponseStarted:
        return "onResponseStarted"_s;
    case WebExtensionEventListenerType::WebRequestOnSendHeaders:
        return "onSendHeaders"_s;
    case WebExtensionEventListenerType::WindowsOnCreated:
        return "onCreated"_s;
    case WebExtensionEventListenerType::WindowsOnFocusChanged:
        return "onFocusChanged"_s;
    case WebExtensionEventListenerType::WindowsOnRemoved:
        return "onRemoved"_s;
    }
}

using WebExtensionEventListenerTypeWorldPair = std::pair<WebExtensionEventListenerType, WebExtensionContentWorldType>;

} // namespace WebKit

namespace WTF {

template<> struct DefaultHash<WebKit::WebExtensionEventListenerType> : IntHash<WebKit::WebExtensionEventListenerType> { };
template<> struct HashTraits<WebKit::WebExtensionEventListenerType> : StrongEnumHashTraits<WebKit::WebExtensionEventListenerType> { };

} // namespace WTF

#endif // ENABLE(WK_WEB_EXTENSIONS)
