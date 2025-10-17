/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 23, 2025.
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

#include "APIClient.h"
#include "APIInjectedBundlePageUIClient.h"
#include "WKBundlePage.h"
#include "WebEvent.h"
#include <wtf/Forward.h>
#include <wtf/TZoneMalloc.h>

namespace API {
class Object;

template<> struct ClientTraits<WKBundlePageUIClientBase> {
    typedef std::tuple<WKBundlePageUIClientV0, WKBundlePageUIClientV1, WKBundlePageUIClientV2, WKBundlePageUIClientV3, WKBundlePageUIClientV4, WKBundlePageUIClientV5> Versions;
};
}

namespace WebKit {

class InjectedBundlePageUIClient : public API::Client<WKBundlePageUIClientBase>, public API::InjectedBundle::PageUIClient {
    WTF_MAKE_TZONE_ALLOCATED(InjectedBundlePageUIClient);
public:
    explicit InjectedBundlePageUIClient(const WKBundlePageUIClientBase*);

    void willAddMessageToConsole(WebPage*, MessageSource, MessageLevel, const String& message, unsigned lineNumber, unsigned columnNumber, const String& sourceID) override;
    void willAddMessageWithArgumentsToConsole(WebPage*, MessageSource, MessageLevel, const String& message, std::span<const String> messageArguments, unsigned lineNumber, unsigned columnNumber, const String& sourceID) override;
    void willRunJavaScriptAlert(WebPage*, const String&, WebFrame*) override;
    void willRunJavaScriptConfirm(WebPage*, const String&, WebFrame*) override;
    void willRunJavaScriptPrompt(WebPage*, const String&, const String&, WebFrame*) override;
    void mouseDidMoveOverElement(WebPage*, const WebCore::HitTestResult&, OptionSet<WebEventModifier>, RefPtr<API::Object>& userData) override;
    void pageDidScroll(WebPage*) override;

    UIElementVisibility statusBarIsVisible(WebPage*) override;
    UIElementVisibility menuBarIsVisible(WebPage*) override;
    UIElementVisibility toolbarsAreVisible(WebPage*) override;

    uint64_t didExceedDatabaseQuota(WebPage*, API::SecurityOrigin*, const String& databaseName, const String& databaseDisplayName, uint64_t currentQuotaBytes, uint64_t currentOriginUsageBytes, uint64_t currentDatabaseUsageBytes, uint64_t expectedUsageBytes) override;

    void didClickAutoFillButton(WebPage&, InjectedBundleNodeHandle&, RefPtr<API::Object>& userData) override;

    void didResignInputElementStrongPasswordAppearance(WebPage&, InjectedBundleNodeHandle&, RefPtr<API::Object>& userData) override;
};

} // namespace WebKit
