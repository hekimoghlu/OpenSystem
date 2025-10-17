/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 8, 2022.
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
#ifndef APIAutomationSessionClient_h
#define APIAutomationSessionClient_h

#include <wtf/CompletionHandler.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/WTFString.h>

namespace WebKit {
class WebAutomationSession;
class WebPageProxy;
}

namespace API {

enum AutomationSessionBrowsingContextOptions : uint16_t {
    AutomationSessionBrowsingContextOptionsPreferNewTab = 1 << 0,
};

#if ENABLE(WK_WEB_EXTENSIONS_IN_WEBDRIVER)
enum AutomationSessionWebExtensionResourceOptions : uint16_t {
    AutomationSessionWebExtensionResourceOptionsPath = 1 << 0,
    AutomationSessionWebExtensionResourceOptionsArchivePath = 1 << 1,
    AutomationSessionWebExtensionResourceOptionsBase64 = 1 << 2,
};
#endif

class AutomationSessionClient {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(AutomationSessionClient);
public:
    enum class JavaScriptDialogType {
        Alert,
        Confirm,
        Prompt,
        BeforeUnloadConfirm
    };

    enum class BrowsingContextPresentation {
        Tab,
        Window,
    };

#if ENABLE(WK_WEB_EXTENSIONS_IN_WEBDRIVER)
    enum class WebExtensionResourceOptions {
        Path,
        ArchivePath,
        Base64,
    };
#endif

    virtual ~AutomationSessionClient() { }

    virtual WTF::String sessionIdentifier() const { return WTF::String(); }
    virtual void didDisconnectFromRemote(WebKit::WebAutomationSession&) { }
    virtual void requestNewPageWithOptions(WebKit::WebAutomationSession&, AutomationSessionBrowsingContextOptions, CompletionHandler<void(WebKit::WebPageProxy*)>&& completionHandler) { completionHandler(nullptr); }
    virtual void requestMaximizeWindowOfPage(WebKit::WebAutomationSession&, WebKit::WebPageProxy&, CompletionHandler<void()>&& completionHandler) { completionHandler(); }
    virtual void requestHideWindowOfPage(WebKit::WebAutomationSession&, WebKit::WebPageProxy&, CompletionHandler<void()>&& completionHandler) { completionHandler(); }
    virtual void requestRestoreWindowOfPage(WebKit::WebAutomationSession&, WebKit::WebPageProxy&, CompletionHandler<void()>&& completionHandler) { completionHandler(); }
    virtual void requestSwitchToPage(WebKit::WebAutomationSession&, WebKit::WebPageProxy&, CompletionHandler<void()>&& completionHandler) { completionHandler(); }
    virtual bool isShowingJavaScriptDialogOnPage(WebKit::WebAutomationSession&, WebKit::WebPageProxy&) { return false; }
    virtual void dismissCurrentJavaScriptDialogOnPage(WebKit::WebAutomationSession&, WebKit::WebPageProxy&) { }
    virtual void acceptCurrentJavaScriptDialogOnPage(WebKit::WebAutomationSession&, WebKit::WebPageProxy&) { }
    virtual WTF::String messageOfCurrentJavaScriptDialogOnPage(WebKit::WebAutomationSession&, WebKit::WebPageProxy&) { return WTF::String(); }
    virtual void setUserInputForCurrentJavaScriptPromptOnPage(WebKit::WebAutomationSession&, WebKit::WebPageProxy&, const WTF::String&) { }
    virtual std::optional<JavaScriptDialogType> typeOfCurrentJavaScriptDialogOnPage(WebKit::WebAutomationSession&, WebKit::WebPageProxy&) { return std::nullopt; }
    virtual BrowsingContextPresentation currentPresentationOfPage(WebKit::WebAutomationSession&, WebKit::WebPageProxy&) { return BrowsingContextPresentation::Window; }
#if ENABLE(WK_WEB_EXTENSIONS_IN_WEBDRIVER)
    virtual void loadWebExtensionWithOptions(WebKit::WebAutomationSession&, API::AutomationSessionWebExtensionResourceOptions, const WTF::String& resource, CompletionHandler<void(const WTF::String&)>&& completionHandler) { completionHandler(WTF::String()); }
    virtual void unloadWebExtension(WebKit::WebAutomationSession&, const WTF::String& identifier, CompletionHandler<void(bool)>&& completionHandler) { completionHandler(false); }
#endif
};

} // namespace API

#endif // APIAutomationSessionClient_h
