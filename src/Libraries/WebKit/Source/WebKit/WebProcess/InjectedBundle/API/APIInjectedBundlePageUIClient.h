/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 22, 2023.
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

#include "WebEvent.h"
#include <JavaScriptCore/ConsoleTypes.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {
class HitTestResult;
}

namespace WebKit {
class InjectedBundleNodeHandle;
class WebFrame;
class WebPage;
}

namespace API {

class Object;
class SecurityOrigin;

namespace InjectedBundle {

class PageUIClient {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(PageUIClient);
public:
    virtual ~PageUIClient() { }

    virtual void willAddMessageToConsole(WebKit::WebPage*, JSC::MessageSource, JSC::MessageLevel, const WTF::String& message, unsigned lineNumber, unsigned columnNumber, const WTF::String& sourceID) { UNUSED_PARAM(message); UNUSED_PARAM(lineNumber); UNUSED_PARAM(columnNumber); UNUSED_PARAM(sourceID); }
    virtual void willAddMessageWithArgumentsToConsole(WebKit::WebPage*, JSC::MessageSource, JSC::MessageLevel, const WTF::String& message, std::span<const WTF::String> messageArguments, unsigned lineNumber, unsigned columnNumber, const WTF::String& sourceID) { UNUSED_PARAM(message); UNUSED_PARAM(messageArguments); UNUSED_PARAM(lineNumber); UNUSED_PARAM(columnNumber); UNUSED_PARAM(sourceID); }
    virtual void willRunJavaScriptAlert(WebKit::WebPage*, const WTF::String&, WebKit::WebFrame*) { }
    virtual void willRunJavaScriptConfirm(WebKit::WebPage*, const WTF::String&, WebKit::WebFrame*) { }
    virtual void willRunJavaScriptPrompt(WebKit::WebPage*, const WTF::String&, const WTF::String&, WebKit::WebFrame*) { }
    virtual void mouseDidMoveOverElement(WebKit::WebPage*, const WebCore::HitTestResult&, OptionSet<WebKit::WebEventModifier>, RefPtr<API::Object>& userData) { UNUSED_PARAM(userData); }
    virtual void pageDidScroll(WebKit::WebPage*) { }

    enum class UIElementVisibility {
        Unknown,
        Visible,
        Hidden,
    };

    virtual UIElementVisibility statusBarIsVisible(WebKit::WebPage*) { return UIElementVisibility::Unknown; }
    virtual UIElementVisibility menuBarIsVisible(WebKit::WebPage*) { return UIElementVisibility::Unknown; }
    virtual UIElementVisibility toolbarsAreVisible(WebKit::WebPage*) { return UIElementVisibility::Unknown; }

    virtual uint64_t didExceedDatabaseQuota(WebKit::WebPage*, SecurityOrigin*, const WTF::String& databaseName, const WTF::String& databaseDisplayName, uint64_t currentQuotaBytes, uint64_t currentOriginUsageBytes, uint64_t currentDatabaseUsageBytes, uint64_t expectedUsageBytes)
    {
        UNUSED_PARAM(databaseName);
        UNUSED_PARAM(databaseDisplayName);
        UNUSED_PARAM(currentQuotaBytes);
        UNUSED_PARAM(currentOriginUsageBytes);
        UNUSED_PARAM(currentDatabaseUsageBytes);
        UNUSED_PARAM(expectedUsageBytes);
        return 0;
    }

    virtual WTF::String plugInStartLabelTitle(const WTF::String& mimeType) const { UNUSED_PARAM(mimeType); return emptyString(); }
    virtual WTF::String plugInStartLabelSubtitle(const WTF::String& mimeType) const { UNUSED_PARAM(mimeType); return emptyString(); }
    virtual WTF::String plugInExtraStyleSheet() const { return emptyString(); }
    virtual WTF::String plugInExtraScript() const { return emptyString(); }

    virtual void didClickAutoFillButton(WebKit::WebPage&, WebKit::InjectedBundleNodeHandle&, RefPtr<API::Object>&) { }

    virtual void didResignInputElementStrongPasswordAppearance(WebKit::WebPage&, WebKit::InjectedBundleNodeHandle&, RefPtr<API::Object>&) { };
};

} // namespace InjectedBundle

} // namespace API
