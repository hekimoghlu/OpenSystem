/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 11, 2023.
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

#include "HardwareAccelerationManager.h"
#include "RemoteWebInspectorUIMessages.h"
#include "WebInspectorUIProxy.h"
#include "WebKitInspectorWindow.h"
#include "WebKitWebViewBasePrivate.h"
#include "WebPageGroup.h"
#include "WebProcessPool.h"
#include <WebCore/CertificateInfo.h>
#include <WebCore/GtkVersioning.h>
#include <wtf/text/Base64.h>

namespace WebKit {
using namespace WebCore;

void RemoteWebInspectorUIProxy::updateWindowTitle(const CString& targetName)
{
    if (!m_window)
        return;
    webkitInspectorWindowSetSubtitle(WEBKIT_INSPECTOR_WINDOW(m_window.get()), !targetName.isNull() ? targetName.data() : nullptr);
}

static void remoteInspectorViewDestroyed(RemoteWebInspectorUIProxy* inspectorProxy)
{
    inspectorProxy->closeFromCrash();
}

WebPageProxy* RemoteWebInspectorUIProxy::platformCreateFrontendPageAndWindow()
{
    ASSERT(!m_webView);

    auto preferences = WebPreferences::create(String(), "WebKit2."_s, "WebKit2."_s);
#if ENABLE(DEVELOPER_MODE)
    // Allow developers to inspect the Web Inspector in debug builds without changing settings.
    preferences->setDeveloperExtrasEnabled(true);
    preferences->setLogsPageMessagesToSystemConsoleEnabled(true);
#endif

    // If hardware acceleration is available and not forced already, force it always for the remote inspector view.
    const auto& hardwareAccelerationManager = HardwareAccelerationManager::singleton();
    if (hardwareAccelerationManager.canUseHardwareAcceleration() && !hardwareAccelerationManager.forceHardwareAcceleration()) {
        preferences->setForceCompositingMode(true);
        preferences->setThreadedScrollingEnabled(true);
    }
    auto pageGroup = WebPageGroup::create(WebKit::defaultInspectorPageGroupIdentifierForPage(nullptr));

    auto pageConfiguration = API::PageConfiguration::create();
    pageConfiguration->setProcessPool(&WebKit::defaultInspectorProcessPool(inspectorLevelForPage(nullptr)));
    pageConfiguration->setPreferences(preferences.ptr());
    pageConfiguration->setPageGroup(pageGroup.ptr());
    m_webView.reset(GTK_WIDGET(webkitWebViewBaseCreate(*pageConfiguration.ptr())));
    g_signal_connect_swapped(m_webView.get(), "destroy", G_CALLBACK(remoteInspectorViewDestroyed), this);

    m_window.reset(webkitInspectorWindowNew());
#if USE(GTK4)
    gtk_window_set_child(GTK_WINDOW(m_window.get()), m_webView.get());
#else
    gtk_container_add(GTK_CONTAINER(m_window.get()), m_webView.get());
    gtk_widget_show(m_webView.get());
#endif

    gtk_window_present(GTK_WINDOW(m_window.get()));

    return webkitWebViewBaseGetPage(WEBKIT_WEB_VIEW_BASE(m_webView.get()));
}

void RemoteWebInspectorUIProxy::platformCloseFrontendPageAndWindow()
{
    if (m_webView) {
        if (auto* webPage = webkitWebViewBaseGetPage(WEBKIT_WEB_VIEW_BASE(m_webView.get())))
            webPage->close();
    }
    if (m_window)
        gtk_widget_destroy(m_window.get());
}

void RemoteWebInspectorUIProxy::platformResetState()
{
}

void RemoteWebInspectorUIProxy::platformBringToFront()
{
    if (m_window)
        gtk_window_present(GTK_WINDOW(m_window.get()));
}

static void remoteFileReplaceContentsCallback(GObject* sourceObject, GAsyncResult* result, gpointer userData)
{
    GFile* file = G_FILE(sourceObject);
    g_file_replace_contents_finish(file, result, nullptr, nullptr);
}

void RemoteWebInspectorUIProxy::platformSave(Vector<InspectorFrontendClient::SaveData>&& saveDatas, bool forceSaveAs)
{
    ASSERT(saveDatas.size() == 1);
    UNUSED_PARAM(forceSaveAs);

    GRefPtr<GtkFileChooserNative> dialog = adoptGRef(gtk_file_chooser_native_new("Save File",
        GTK_WINDOW(m_window.get()), GTK_FILE_CHOOSER_ACTION_SAVE, "Save", "Cancel"));

    GtkFileChooser* chooser = GTK_FILE_CHOOSER(dialog.get());
#if !USE(GTK4)
    gtk_file_chooser_set_do_overwrite_confirmation(chooser, TRUE);
#endif

    // Some inspector views (Audits for instance) use a custom URI scheme, such
    // as web-inspector. So we can't rely on the URL being a valid file:/// URL
    // unfortunately.
    URL url { saveDatas[0].url };
    // Strip leading / character.
    gtk_file_chooser_set_current_name(chooser, url.path().substring(1).utf8().data());

    if (gtk_native_dialog_run(GTK_NATIVE_DIALOG(dialog.get())) != GTK_RESPONSE_ACCEPT)
        return;

    Vector<uint8_t> dataVector;
    CString dataString;
    if (saveDatas[0].base64Encoded) {
        auto decodedData = base64Decode(saveDatas[0].content, { Base64DecodeOption::ValidatePadding });
        if (!decodedData)
            return;
        decodedData->shrinkToFit();
        dataVector = WTFMove(*decodedData);
    } else
        dataString = saveDatas[0].content.utf8();

    const char* data = !dataString.isNull() ? dataString.data() : reinterpret_cast<char*>(dataVector.data());
    size_t dataLength = !dataString.isNull() ? dataString.length() : dataVector.size();
    GRefPtr<GFile> file = adoptGRef(gtk_file_chooser_get_file(chooser));
    GUniquePtr<char> path(g_file_get_path(file.get()));
    g_file_replace_contents_async(file.get(), data, dataLength, nullptr, false,
        G_FILE_CREATE_REPLACE_DESTINATION, nullptr, remoteFileReplaceContentsCallback, protectedInspectorPage().get());
}

void RemoteWebInspectorUIProxy::platformLoad(const String&, CompletionHandler<void(const String&)>&& completionHandler)
{
    completionHandler(nullString());
}

void RemoteWebInspectorUIProxy::platformPickColorFromScreen(CompletionHandler<void(const std::optional<WebCore::Color>&)>&& completionHandler)
{
    completionHandler({ });
}

void RemoteWebInspectorUIProxy::platformSetSheetRect(const FloatRect&)
{
}

void RemoteWebInspectorUIProxy::platformSetForcedAppearance(InspectorFrontendClient::Appearance)
{
}

void RemoteWebInspectorUIProxy::platformStartWindowDrag()
{
}

void RemoteWebInspectorUIProxy::platformOpenURLExternally(const String&)
{
}

void RemoteWebInspectorUIProxy::platformRevealFileExternally(const String&)
{
}

void RemoteWebInspectorUIProxy::platformShowCertificate(const CertificateInfo&)
{
}

} // namespace WebKit

#endif // ENABLE(REMOTE_INSPECTOR)
