/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 11, 2023.
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
#include "RemoteWebInspectorUIProxy.h"

#if ENABLE(REMOTE_INSPECTOR)

#include "APIPageConfiguration.h"
#include "InspectorResourceURLSchemeHandler.h"
#include "WebInspectorUIProxy.h"
#include "WebPageGroup.h"
#include "WebPageProxy.h"
#include "WebProcessPool.h"
#include "WebView.h"
#include <WebCore/InspectorFrontendClient.h>
#include <WebCore/IntRect.h>
#include <wtf/FileSystem.h>

namespace WebKit {

static LPCTSTR RemoteWebInspectorUIProxyPointerProp = TEXT("RemoteWebInspectorUIProxyPointer");
const LPCWSTR RemoteWebInspectorUIProxyClassName = L"RemoteWebInspectorUIProxyClass";

LRESULT CALLBACK RemoteWebInspectorUIProxy::WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    RemoteWebInspectorUIProxy* client = reinterpret_cast<RemoteWebInspectorUIProxy*>(::GetProp(hwnd, RemoteWebInspectorUIProxyPointerProp));

    switch (msg) {
    case WM_SIZE:
        return client->sizeChange();
    case WM_CLOSE:
        return client->onClose();
    default:
        break;
    }

    return ::DefWindowProc(hwnd, msg, wParam, lParam);
}

static ATOM registerWindowClass()
{
    static bool haveRegisteredWindowClass = false;
    if (haveRegisteredWindowClass)
        return true;

    WNDCLASSEX wcex;
    wcex.cbSize = sizeof(WNDCLASSEX);
    wcex.style          = 0;
    wcex.lpfnWndProc    = RemoteWebInspectorUIProxy::WndProc;
    wcex.cbClsExtra     = 0;
    wcex.cbWndExtra     = 0;
    wcex.hInstance      = 0;
    wcex.hIcon          = 0;
    wcex.hCursor        = LoadCursor(0, IDC_ARROW);
    wcex.hbrBackground  = 0;
    wcex.lpszMenuName   = 0;
    wcex.lpszClassName  = RemoteWebInspectorUIProxyClassName;
    wcex.hIconSm        = 0;

    haveRegisteredWindowClass = true;
    return ::RegisterClassEx(&wcex);
}

LRESULT RemoteWebInspectorUIProxy::sizeChange()
{
    if (!m_webView)
        return 0;

    RECT rect;
    ::GetClientRect(m_frontendHandle, &rect);
    ::SetWindowPos(m_webView->window(), 0, rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top, SWP_NOZORDER);
    return 0;
}

LRESULT RemoteWebInspectorUIProxy::onClose()
{
    ::ShowWindow(m_frontendHandle, SW_HIDE);
    frontendDidClose();
    return 0;
}

WebPageProxy* RemoteWebInspectorUIProxy::platformCreateFrontendPageAndWindow()
{
    RefPtr<WebPreferences> preferences = WebPreferences::create(String(), "WebKit2."_s, "WebKit2."_s);

#if ENABLE(DEVELOPER_MODE)
    preferences->setDeveloperExtrasEnabled(true);
    preferences->setLogsPageMessagesToSystemConsoleEnabled(true);
#endif

    RefPtr<WebPageGroup> pageGroup = WebPageGroup::create(WebKit::defaultInspectorPageGroupIdentifierForPage(nullptr));

    auto pageConfiguration = API::PageConfiguration::create();
    pageConfiguration->setProcessPool(&WebKit::defaultInspectorProcessPool(inspectorLevelForPage(nullptr)));
    pageConfiguration->setPreferences(preferences.get());
    pageConfiguration->setPageGroup(pageGroup.get());

    WebCore::IntRect rect(60, 200, 1500, 1000);
    registerWindowClass();
    m_frontendHandle = ::CreateWindowEx(0, RemoteWebInspectorUIProxyClassName, 0, WS_OVERLAPPEDWINDOW,
        rect.x(), rect.y(), rect.width(), rect.height(), 0, 0, 0, 0);

    ::SetProp(m_frontendHandle, RemoteWebInspectorUIProxyPointerProp, reinterpret_cast<HANDLE>(this));
    ShowWindow(m_frontendHandle, SW_SHOW);

    RECT r;
    ::GetClientRect(m_frontendHandle, &r);
    m_webView = WebView::create(r, pageConfiguration, m_frontendHandle);

    auto inspectorPage = m_webView->page();
    inspectorPage->setURLSchemeHandlerForScheme(InspectorResourceURLSchemeHandler::create(), "inspector-resource"_s);

    return inspectorPage;
}

void RemoteWebInspectorUIProxy::platformSave(Vector<WebCore::InspectorFrontendClient::SaveData>&& saveDatas, bool /* forceSaveAs */)
{
    // Currently, file saving is only possible with SaveMode::SingleFile.
    // This is determined in RemoteWebInspectorUI::canSave().
    ASSERT(saveDatas.size() == 1);
    WebInspectorUIProxy::showSavePanelForSingleFile(m_frontendHandle, WTFMove(saveDatas));
}

void RemoteWebInspectorUIProxy::platformResetState() { }
void RemoteWebInspectorUIProxy::platformBringToFront() { }
void RemoteWebInspectorUIProxy::platformLoad(const String&, CompletionHandler<void(const String&)>&& completionHandler) { completionHandler(nullString()); }
void RemoteWebInspectorUIProxy::platformPickColorFromScreen(CompletionHandler<void(const std::optional<WebCore::Color>&)>&& completionHandler) { completionHandler({ }); }
void RemoteWebInspectorUIProxy::platformSetSheetRect(const WebCore::FloatRect&) { }
void RemoteWebInspectorUIProxy::platformSetForcedAppearance(WebCore::InspectorFrontendClient::Appearance) { }
void RemoteWebInspectorUIProxy::platformStartWindowDrag() { }
void RemoteWebInspectorUIProxy::platformOpenURLExternally(const String&) { }
void RemoteWebInspectorUIProxy::platformRevealFileExternally(const String&) { }
void RemoteWebInspectorUIProxy::platformShowCertificate(const WebCore::CertificateInfo&) { }

void RemoteWebInspectorUIProxy::platformCloseFrontendPageAndWindow()
{
    ::DestroyWindow(m_frontendHandle);
}

} // namespace WebKit

#endif // ENABLE(REMOTE_INSPECTOR)
