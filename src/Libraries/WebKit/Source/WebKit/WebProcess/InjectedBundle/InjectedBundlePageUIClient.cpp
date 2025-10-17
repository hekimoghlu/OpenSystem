/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 22, 2022.
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
#include "InjectedBundlePageUIClient.h"

#include "APIArray.h"
#include "APISecurityOrigin.h"
#include "InjectedBundleHitTestResult.h"
#include "InjectedBundleNodeHandle.h"
#include "WKAPICast.h"
#include "WKBundleAPICast.h"
#include "WebFrame.h"
#include "WebPage.h"
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/WTFString.h>

namespace WebKit {
using namespace WebCore;

WTF_MAKE_TZONE_ALLOCATED_IMPL(InjectedBundlePageUIClient);

InjectedBundlePageUIClient::InjectedBundlePageUIClient(const WKBundlePageUIClientBase* client)
{
    initialize(client);
}

void InjectedBundlePageUIClient::willAddMessageToConsole(WebPage* page, MessageSource, MessageLevel, const String& message, unsigned lineNumber, unsigned /*columnNumber*/, const String& /*sourceID*/)
{
    if (m_client.willAddMessageToConsole)
        m_client.willAddMessageToConsole(toAPI(page), toAPI(message.impl()), lineNumber, m_client.base.clientInfo);
}

void InjectedBundlePageUIClient::willAddMessageWithArgumentsToConsole(WebPage* page, MessageSource, MessageLevel, const String& message, std::span<const String> messageArguments, unsigned lineNumber, unsigned columnNumber, const String& sourceID)
{
    if (m_client.willAddMessageWithDetailsToConsole)
        m_client.willAddMessageWithDetailsToConsole(toAPI(page), toAPI(message.impl()), toAPI(&API::Array::createStringArray(messageArguments).leakRef()), lineNumber, columnNumber, toAPI(sourceID.impl()), m_client.base.clientInfo);
}

void InjectedBundlePageUIClient::willRunJavaScriptAlert(WebPage* page, const String& alertText, WebFrame* frame)
{
    if (m_client.willRunJavaScriptAlert)
        m_client.willRunJavaScriptAlert(toAPI(page), toAPI(alertText.impl()), toAPI(frame), m_client.base.clientInfo);
}

void InjectedBundlePageUIClient::willRunJavaScriptConfirm(WebPage* page, const String& message, WebFrame* frame)
{
    if (m_client.willRunJavaScriptConfirm)
        m_client.willRunJavaScriptConfirm(toAPI(page), toAPI(message.impl()), toAPI(frame), m_client.base.clientInfo);
}

void InjectedBundlePageUIClient::willRunJavaScriptPrompt(WebPage* page, const String& message, const String& defaultValue, WebFrame* frame)
{
    if (m_client.willRunJavaScriptPrompt)
        m_client.willRunJavaScriptPrompt(toAPI(page), toAPI(message.impl()), toAPI(defaultValue.impl()), toAPI(frame), m_client.base.clientInfo);
}

void InjectedBundlePageUIClient::mouseDidMoveOverElement(WebPage* page, const HitTestResult& coreHitTestResult, OptionSet<WebEventModifier> modifiers, RefPtr<API::Object>& userData)
{
    if (!m_client.mouseDidMoveOverElement)
        return;

    auto hitTestResult = InjectedBundleHitTestResult::create(coreHitTestResult);

    WKTypeRef userDataToPass = 0;
    m_client.mouseDidMoveOverElement(toAPI(page), toAPI(hitTestResult.ptr()), toAPI(modifiers), &userDataToPass, m_client.base.clientInfo);
    userData = adoptRef(toImpl(userDataToPass));
}

void InjectedBundlePageUIClient::pageDidScroll(WebPage* page)
{
    if (!m_client.pageDidScroll)
        return;

    m_client.pageDidScroll(toAPI(page), m_client.base.clientInfo);
}

static API::InjectedBundle::PageUIClient::UIElementVisibility toUIElementVisibility(WKBundlePageUIElementVisibility visibility)
{
    switch (visibility) {
    case WKBundlePageUIElementVisibilityUnknown:
        return API::InjectedBundle::PageUIClient::UIElementVisibility::Unknown;
    case WKBundlePageUIElementVisible:
        return API::InjectedBundle::PageUIClient::UIElementVisibility::Visible;
    case WKBundlePageUIElementHidden:
        return API::InjectedBundle::PageUIClient::UIElementVisibility::Hidden;
    }

    ASSERT_NOT_REACHED();
    return API::InjectedBundle::PageUIClient::UIElementVisibility::Unknown;
}

API::InjectedBundle::PageUIClient::UIElementVisibility InjectedBundlePageUIClient::statusBarIsVisible(WebPage* page)
{
    if (!m_client.statusBarIsVisible)
        return API::InjectedBundle::PageUIClient::UIElementVisibility::Unknown;
    
    return toUIElementVisibility(m_client.statusBarIsVisible(toAPI(page), m_client.base.clientInfo));
}

API::InjectedBundle::PageUIClient::UIElementVisibility InjectedBundlePageUIClient::menuBarIsVisible(WebPage* page)
{
    if (!m_client.menuBarIsVisible)
        return API::InjectedBundle::PageUIClient::UIElementVisibility::Unknown;
    
    return toUIElementVisibility(m_client.menuBarIsVisible(toAPI(page), m_client.base.clientInfo));
}

API::InjectedBundle::PageUIClient::UIElementVisibility InjectedBundlePageUIClient::toolbarsAreVisible(WebPage* page)
{
    if (!m_client.toolbarsAreVisible)
        return API::InjectedBundle::PageUIClient::UIElementVisibility::Unknown;
    
    return toUIElementVisibility(m_client.toolbarsAreVisible(toAPI(page), m_client.base.clientInfo));
}

uint64_t InjectedBundlePageUIClient::didExceedDatabaseQuota(WebPage* page, API::SecurityOrigin* origin, const String& databaseName, const String& databaseDisplayName, uint64_t currentQuotaBytes, uint64_t currentOriginUsageBytes, uint64_t currentDatabaseUsageBytes, uint64_t expectedUsageBytes)
{
    if (!m_client.didExceedDatabaseQuota)
        return 0;

    return m_client.didExceedDatabaseQuota(toAPI(page), toAPI(origin), toAPI(databaseName.impl()), toAPI(databaseDisplayName.impl()), currentQuotaBytes, currentOriginUsageBytes, currentDatabaseUsageBytes, expectedUsageBytes, m_client.base.clientInfo);
}

void InjectedBundlePageUIClient::didClickAutoFillButton(WebPage& page, InjectedBundleNodeHandle& nodeHandle, RefPtr<API::Object>& userData)
{
    if (!m_client.didClickAutoFillButton)
        return;

    WKTypeRef userDataToPass = nullptr;
    m_client.didClickAutoFillButton(toAPI(&page), toAPI(&nodeHandle), &userDataToPass, m_client.base.clientInfo);
    userData = adoptRef(toImpl(userDataToPass));
}

void InjectedBundlePageUIClient::didResignInputElementStrongPasswordAppearance(WebPage& page, InjectedBundleNodeHandle& nodeHandle, RefPtr<API::Object>& userData)
{
    if (!m_client.didResignInputElementStrongPasswordAppearance)
        return;

    WKTypeRef userDataToPass = nullptr;
    m_client.didResignInputElementStrongPasswordAppearance(toAPI(&page), toAPI(&nodeHandle), &userDataToPass, m_client.base.clientInfo);
    userData = adoptRef(toImpl(userDataToPass));
}

} // namespace WebKit
