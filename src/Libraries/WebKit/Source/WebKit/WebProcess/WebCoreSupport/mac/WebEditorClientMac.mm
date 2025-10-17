/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 25, 2024.
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
#import "config.h"
#import "WebEditorClient.h"

#if PLATFORM(MAC)

#import "MessageSenderInlines.h"
#import "TextCheckerState.h"
#import "WebPage.h"
#import "WebPageProxyMessages.h"
#import "WebProcess.h"
#import <WebCore/Editor.h>
#import <WebCore/FocusController.h>
#import <WebCore/KeyboardEvent.h>
#import <WebCore/LocalFrame.h>
#import <WebCore/NotImplemented.h>
#import <WebCore/Page.h>
#import <wtf/cocoa/NSURLExtras.h>

namespace WebKit {
using namespace WebCore;
    
void WebEditorClient::handleKeyboardEvent(KeyboardEvent& event)
{
    if (m_page->handleEditingKeyboardEvent(event))
        event.setDefaultHandled();
}

void WebEditorClient::handleInputMethodKeydown(KeyboardEvent& event)
{
    if (event.handledByInputMethod())
        event.setDefaultHandled();
}

void WebEditorClient::setInsertionPasteboard(const String&)
{
    // This is used only by Mail, no need to implement it now.
    notImplemented();
}

static void changeWordCase(WebPage* page, NSString *(*changeCase)(NSString *))
{
    RefPtr frame = page->corePage()->checkedFocusController()->focusedOrMainFrame();
    if (!frame)
        return;
    if (!frame->editor().canEdit())
        return;

    frame->editor().command("selectWord"_s).execute();

    NSString *selectedString = frame->displayStringModifiedByEncoding(frame->editor().selectedText());
    page->replaceSelectionWithText(frame.get(), changeCase(selectedString));
}

void WebEditorClient::uppercaseWord()
{
    changeWordCase(RefPtr { m_page.get() }.get(), [] (NSString *string) {
        return [string uppercaseString];
    });
}

void WebEditorClient::lowercaseWord()
{
    changeWordCase(RefPtr { m_page.get() }.get(), [] (NSString *string) {
        return [string lowercaseString];
    });
}

void WebEditorClient::capitalizeWord()
{
    changeWordCase(RefPtr { m_page.get() }.get(), [] (NSString *string) {
        return [string capitalizedString];
    });
}

#if USE(AUTOMATIC_TEXT_REPLACEMENT)

void WebEditorClient::showSubstitutionsPanel(bool)
{
    notImplemented();
}

bool WebEditorClient::substitutionsPanelIsShowing()
{
    auto sendResult = Ref { *m_page }->sendSync(Messages::WebPageProxy::SubstitutionsPanelIsShowing());
    auto [isShowing] = sendResult.takeReplyOr(false);
    return isShowing;
}

void WebEditorClient::toggleSmartInsertDelete()
{
    Ref { *m_page }->send(Messages::WebPageProxy::toggleSmartInsertDelete());
}

bool WebEditorClient::isAutomaticQuoteSubstitutionEnabled()
{
    if (Ref { *m_page }->isControlledByAutomation())
        return false;

    return WebProcess::singleton().textCheckerState().contains(TextCheckerState::AutomaticQuoteSubstitutionEnabled);
}

void WebEditorClient::toggleAutomaticQuoteSubstitution()
{
    Ref { *m_page }->send(Messages::WebPageProxy::toggleAutomaticQuoteSubstitution());
}

bool WebEditorClient::isAutomaticLinkDetectionEnabled()
{
    return WebProcess::singleton().textCheckerState().contains(TextCheckerState::AutomaticLinkDetectionEnabled);
}

void WebEditorClient::toggleAutomaticLinkDetection()
{
    Ref { *m_page }->send(Messages::WebPageProxy::toggleAutomaticLinkDetection());
}

bool WebEditorClient::isAutomaticDashSubstitutionEnabled()
{
    if (m_page->isControlledByAutomation())
        return false;

    return WebProcess::singleton().textCheckerState().contains(TextCheckerState::AutomaticDashSubstitutionEnabled);
}

void WebEditorClient::toggleAutomaticDashSubstitution()
{
    Ref { *m_page }->send(Messages::WebPageProxy::toggleAutomaticDashSubstitution());
}

bool WebEditorClient::isAutomaticTextReplacementEnabled()
{
    if (m_page->isControlledByAutomation())
        return false;

    return WebProcess::singleton().textCheckerState().contains(TextCheckerState::AutomaticTextReplacementEnabled);
}

void WebEditorClient::toggleAutomaticTextReplacement()
{
    Ref { *m_page }->send(Messages::WebPageProxy::toggleAutomaticTextReplacement());
}

bool WebEditorClient::isAutomaticSpellingCorrectionEnabled()
{
    if (m_page->isControlledByAutomation())
        return false;

    return WebProcess::singleton().textCheckerState().contains(TextCheckerState::AutomaticSpellingCorrectionEnabled);
}

void WebEditorClient::toggleAutomaticSpellingCorrection()
{
    notImplemented();
}

#endif // USE(AUTOMATIC_TEXT_REPLACEMENT)

} // namespace WebKit

#endif // PLATFORM(MAC)
