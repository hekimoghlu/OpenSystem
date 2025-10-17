/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 11, 2024.
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

#include "WebPage.h"
#include <WebCore/EditorClient.h>
#include <WebCore/TextCheckerClient.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>

namespace WebCore {
#if ENABLE(ATTACHMENT_ELEMENT)
enum class AttachmentAssociatedElementType : uint8_t;
#endif
enum class DOMPasteAccessCategory : uint8_t;
enum class DOMPasteAccessResponse : uint8_t;
}

namespace WebKit {

class WebPage;

class WebEditorClient final : public WebCore::EditorClient, public WebCore::TextCheckerClient {
    WTF_MAKE_TZONE_ALLOCATED(WebEditorClient);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(WebEditorClient);
public:
    WebEditorClient(WebPage* page)
        : m_page(page)
    {
    }

private:
    bool shouldDeleteRange(const std::optional<WebCore::SimpleRange>&) final;
    bool smartInsertDeleteEnabled() final;
    bool isSelectTrailingWhitespaceEnabled() const final;
    bool isContinuousSpellCheckingEnabled() final;
    void toggleContinuousSpellChecking() final;
    bool isGrammarCheckingEnabled() final;
    void toggleGrammarChecking() final;
    int spellCheckerDocumentTag() final;
    
    bool shouldBeginEditing(const WebCore::SimpleRange&) final;
    bool shouldEndEditing(const WebCore::SimpleRange&) final;
    bool shouldInsertNode(WebCore::Node&, const std::optional<WebCore::SimpleRange>&, WebCore::EditorInsertAction) final;
    bool shouldInsertText(const String&, const std::optional<WebCore::SimpleRange>&, WebCore::EditorInsertAction) final;
    bool shouldChangeSelectedRange(const std::optional<WebCore::SimpleRange>& fromRange, const std::optional<WebCore::SimpleRange>& toRange, WebCore::Affinity, bool stillSelecting) final;
    
    bool shouldApplyStyle(const WebCore::StyleProperties&, const std::optional<WebCore::SimpleRange>&) final;
    void didApplyStyle() final;
    bool shouldMoveRangeAfterDelete(const WebCore::SimpleRange&, const WebCore::SimpleRange&) final;

#if ENABLE(ATTACHMENT_ELEMENT)
    void registerAttachmentIdentifier(const String&, const String& contentType, const String& preferredFileName, Ref<WebCore::FragmentedSharedBuffer>&&) final;
    void registerAttachmentIdentifier(const String&, const String& contentType, const String& filePath) final;
    void registerAttachmentIdentifier(const String&) final;
    void registerAttachments(Vector<WebCore::SerializedAttachmentData>&&) final;
    void cloneAttachmentData(const String& fromIdentifier, const String& toIdentifier) final;
    void didInsertAttachmentWithIdentifier(const String& identifier, const String& source, WebCore::AttachmentAssociatedElementType) final;
    void didRemoveAttachmentWithIdentifier(const String& identifier) final;
    bool supportsClientSideAttachmentData() const final { return true; }
    Vector<WebCore::SerializedAttachmentData> serializedAttachmentDataForIdentifiers(const Vector<String>&) final;
#endif

    void didBeginEditing() final;
    void respondToChangedContents() final;
    void respondToChangedSelection(WebCore::LocalFrame*) final;
    void didEndUserTriggeredSelectionChanges() final;
    void updateEditorStateAfterLayoutIfEditabilityChanged() final;
    void discardedComposition(const WebCore::Document&) final;
    void canceledComposition() final;
    void didUpdateComposition() final;
    void didEndEditing() final;
    void willWriteSelectionToPasteboard(const std::optional<WebCore::SimpleRange>&) final;
    void didWriteSelectionToPasteboard() final;
    void getClientPasteboardData(const std::optional<WebCore::SimpleRange>&, Vector<std::pair<String, RefPtr<WebCore::SharedBuffer>>>& pasteboardTypesAndData) final;
    
    void registerUndoStep(WebCore::UndoStep&) final;
    void registerRedoStep(WebCore::UndoStep&) final;
    void clearUndoRedoOperations() final;

    WebCore::DOMPasteAccessResponse requestDOMPasteAccess(WebCore::DOMPasteAccessCategory, WebCore::FrameIdentifier, const String& originIdentifier) final;

    bool canCopyCut(WebCore::LocalFrame*, bool defaultValue) const final;
    bool canPaste(WebCore::LocalFrame*, bool defaultValue) const final;
    bool canUndo() const final;
    bool canRedo() const final;
    
    void undo() final;
    void redo() final;

    void handleKeyboardEvent(WebCore::KeyboardEvent&) final;
    void handleInputMethodKeydown(WebCore::KeyboardEvent&) final;
    
    void textFieldDidBeginEditing(WebCore::Element&) final;
    void textFieldDidEndEditing(WebCore::Element&) final;
    void textDidChangeInTextField(WebCore::Element&) final;
    bool doTextFieldCommandFromEvent(WebCore::Element&, WebCore::KeyboardEvent*) final;
    void textWillBeDeletedInTextField(WebCore::Element&) final;
    void textDidChangeInTextArea(WebCore::Element&) final;
    void overflowScrollPositionChanged() final;
    void subFrameScrollPositionChanged() final;

#if PLATFORM(COCOA)
    void setInsertionPasteboard(const String& pasteboardName) final;
#endif

#if USE(APPKIT)
    void uppercaseWord() final;
    void lowercaseWord() final;
    void capitalizeWord() final;
#endif

#if USE(AUTOMATIC_TEXT_REPLACEMENT)
    void showSubstitutionsPanel(bool show) final;
    bool substitutionsPanelIsShowing() final;
    void toggleSmartInsertDelete() final;
    bool isAutomaticQuoteSubstitutionEnabled() final;
    void toggleAutomaticQuoteSubstitution() final;
    bool isAutomaticLinkDetectionEnabled() final;
    void toggleAutomaticLinkDetection() final;
    bool isAutomaticDashSubstitutionEnabled() final;
    void toggleAutomaticDashSubstitution() final;
    bool isAutomaticTextReplacementEnabled() final;
    void toggleAutomaticTextReplacement() final;
    bool isAutomaticSpellingCorrectionEnabled() final;
    void toggleAutomaticSpellingCorrection() final;
#endif

#if PLATFORM(GTK)
    bool executePendingEditorCommands(WebCore::LocalFrame&, const Vector<WTF::String>&, bool);
    bool handleGtkEditorCommand(WebCore::LocalFrame&, const String& command, bool);
    void getEditorCommandsForKeyEvent(const WebCore::KeyboardEvent*, Vector<WTF::String>&);
    void updateGlobalSelection(WebCore::LocalFrame*);
#endif

    TextCheckerClient* textChecker() final { return this; }

    bool shouldEraseMarkersAfterChangeSelection(WebCore::TextCheckingType) const final;
    void ignoreWordInSpellDocument(const String&) final;
    void learnWord(const String&) final;
    void checkSpellingOfString(StringView, int* misspellingLocation, int* misspellingLength) final;
    void checkGrammarOfString(StringView, Vector<WebCore::GrammarDetail>&, int* badGrammarLocation, int* badGrammarLength) final;

#if USE(UNIFIED_TEXT_CHECKING)
    Vector<WebCore::TextCheckingResult> checkTextOfParagraph(StringView, OptionSet<WebCore::TextCheckingType> checkingTypes, const WebCore::VisibleSelection& currentSelection) final;
#endif

    void updateSpellingUIWithGrammarString(const String&, const WebCore::GrammarDetail&) final;
    void updateSpellingUIWithMisspelledWord(const String&) final;
    void showSpellingUI(bool show) final;
    bool spellingUIIsShowing() final;
    void getGuessesForWord(const String& word, const String& context, const WebCore::VisibleSelection& currentSelection, Vector<String>& guesses) final;
    void setInputMethodState(WebCore::Element*) final;
    void requestCheckingOfString(WebCore::TextCheckingRequest&, const WebCore::VisibleSelection& currentSelection) final;

#if PLATFORM(GTK)
    bool shouldShowUnicodeMenu() final;
#endif

#if PLATFORM(GTK) || PLATFORM(WPE)
    void didDispatchInputMethodKeydown(WebCore::KeyboardEvent&) final;
#endif

#if PLATFORM(IOS_FAMILY)
    void startDelayingAndCoalescingContentChangeNotifications() final;
    void stopDelayingAndCoalescingContentChangeNotifications() final;
    bool hasRichlyEditableSelection() final;
    int getPasteboardItemsCount() final;
    RefPtr<WebCore::DocumentFragment> documentFragmentFromDelegate(int index) final;
    bool performsTwoStepPaste(WebCore::DocumentFragment*) final;
    void updateStringForFind(const String&) final;
    bool shouldAllowSingleClickToChangeSelection(WebCore::Node&, const WebCore::VisibleSelection&) const final;
    bool shouldRevealCurrentSelectionAfterInsertion() const final;
    bool shouldSuppressPasswordEcho() const final;
    bool shouldRemoveDictationAlternativesAfterEditing() const final;
#endif

    void willChangeSelectionForAccessibility() final;
    void didChangeSelectionForAccessibility() final;

    bool performTwoStepDrop(WebCore::DocumentFragment&, const WebCore::SimpleRange&, bool isMove) final;
    bool supportsGlobalSelection() final;

#if PLATFORM(IOS_FAMILY)
    bool shouldDrawVisuallyContiguousBidiSelection() const final;
#endif

    WeakPtr<WebPage> m_page;
};

} // namespace WebKit
