/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 21, 2025.
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

#if PLATFORM(IOS_FAMILY)

#import "WebPage.h"
#import <WebCore/DocumentFragment.h>
#import <WebCore/KeyboardEvent.h>
#import <WebCore/NotImplemented.h>

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

void WebEditorClient::startDelayingAndCoalescingContentChangeNotifications()
{
    notImplemented();
}

void WebEditorClient::stopDelayingAndCoalescingContentChangeNotifications()
{
    notImplemented();
}

bool WebEditorClient::hasRichlyEditableSelection()
{
    return m_page->hasRichlyEditableSelection();
}

int WebEditorClient::getPasteboardItemsCount()
{
    notImplemented();
    return 0;
}

RefPtr<WebCore::DocumentFragment> WebEditorClient::documentFragmentFromDelegate(int)
{
    notImplemented();
    return nullptr;
}

bool WebEditorClient::performsTwoStepPaste(WebCore::DocumentFragment*)
{
    notImplemented();
    return false;
}

void WebEditorClient::updateStringForFind(const String& findString)
{
    m_page->updateStringForFind(findString);
}

void WebEditorClient::overflowScrollPositionChanged()
{
    m_page->didScrollSelection();
}

void WebEditorClient::subFrameScrollPositionChanged()
{
    m_page->didScrollSelection();
}

bool WebEditorClient::shouldAllowSingleClickToChangeSelection(WebCore::Node& targetNode, const WebCore::VisibleSelection& newSelection) const
{
    return m_page->shouldAllowSingleClickToChangeSelection(targetNode, newSelection);
}

bool WebEditorClient::shouldRevealCurrentSelectionAfterInsertion() const
{
    return m_page->shouldRevealCurrentSelectionAfterInsertion();
}

bool WebEditorClient::shouldSuppressPasswordEcho() const
{
    return m_page->screenIsBeingCaptured() || m_page->hardwareKeyboardIsAttached();
}

bool WebEditorClient::shouldRemoveDictationAlternativesAfterEditing() const
{
    return m_page->shouldRemoveDictationAlternativesAfterEditing();
}

bool WebEditorClient::shouldDrawVisuallyContiguousBidiSelection() const
{
    return m_page->shouldDrawVisuallyContiguousBidiSelection();
}

} // namespace WebKit

#endif // PLATFORM(IOS_FAMILY)
