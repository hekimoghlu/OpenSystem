/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 5, 2023.
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
#import <WebCore/DOMPasteAccess.h>
#import <WebCore/EditorClient.h>
#import <WebCore/TextCheckerClient.h>
#import <WebCore/VisibleSelection.h>
#import <wtf/Forward.h>
#import <wtf/Ref.h>
#import <wtf/RetainPtr.h>
#import <wtf/TZoneMalloc.h>
#import <wtf/Vector.h>
#import <wtf/WeakPtr.h>
#import <wtf/text/StringView.h>

#if PLATFORM(IOS_FAMILY)
#import <WebCore/WAKAppKitStubs.h>
#endif

@class WebView;
@class WebEditorUndoTarget;

class WebEditorClient final : public WebCore::EditorClient, public WebCore::TextCheckerClient {
    WTF_MAKE_TZONE_ALLOCATED(WebEditorClient);
public:
    WebEditorClient(WebView *);
    virtual ~WebEditorClient();

    void didCheckSucceed(WebCore::TextCheckingRequestIdentifier, NSArray *results);

private:
    bool isGrammarCheckingEnabled() final;
    void toggleGrammarChecking() final;
    bool isContinuousSpellCheckingEnabled() final;
    void toggleContinuousSpellChecking() final;
    int spellCheckerDocumentTag() final;

    bool smartInsertDeleteEnabled() final;
    bool isSelectTrailingWhitespaceEnabled() const final;

    bool shouldDeleteRange(const std::optional<WebCore::SimpleRange>&) final;

    bool shouldBeginEditing(const WebCore::SimpleRange&) final;
    bool shouldEndEditing(const WebCore::SimpleRange&) final;
    bool shouldInsertNode(WebCore::Node&, const std::optional<WebCore::SimpleRange>&, WebCore::EditorInsertAction) final;
    bool shouldInsertText(const String&, const std::optional<WebCore::SimpleRange>&, WebCore::EditorInsertAction) final;
    bool shouldChangeSelectedRange(const std::optional<WebCore::SimpleRange>& fromRange, const std::optional<WebCore::SimpleRange>& toRange, WebCore::Affinity, bool stillSelecting) final;

    bool shouldApplyStyle(const WebCore::StyleProperties&, const std::optional<WebCore::SimpleRange>&) final;
    void didApplyStyle() final;

    bool shouldMoveRangeAfterDelete(const WebCore::SimpleRange&, const WebCore::SimpleRange& rangeToBeReplaced) final;

    void didBeginEditing() final;
    void didEndEditing() final;
    void willWriteSelectionToPasteboard(const std::optional<WebCore::SimpleRange>&) final;
    void didWriteSelectionToPasteboard() final;
    void getClientPasteboardData(const std::optional<WebCore::SimpleRange>&, Vector<std::pair<String, RefPtr<WebCore::SharedBuffer>>>& pasteboardTypesAndData) final;

    void setInsertionPasteboard(const String&) final;
    WebCore::DOMPasteAccessResponse requestDOMPasteAccess(WebCore::DOMPasteAccessCategory, WebCore::FrameIdentifier, const String&) final { return WebCore::DOMPasteAccessResponse::DeniedForGesture; }

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

    TextCheckerClient* textChecker() final { return this; }

    void respondToChangedContents() final;
    void respondToChangedSelection(WebCore::LocalFrame*) final;
    void didEndUserTriggeredSelectionChanges() final { }
    void updateEditorStateAfterLayoutIfEditabilityChanged() final;
    void discardedComposition(const WebCore::Document&) final;
    void canceledComposition() final;
    void didUpdateComposition() final { }

    void registerUndoStep(WebCore::UndoStep&) final;
    void registerRedoStep(WebCore::UndoStep&) final;
    void clearUndoRedoOperations() final;

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
    void overflowScrollPositionChanged() final { };
    void subFrameScrollPositionChanged() final { };

#if PLATFORM(IOS_FAMILY)
    void startDelayingAndCoalescingContentChangeNotifications() final;
    void stopDelayingAndCoalescingContentChangeNotifications() final;
    bool hasRichlyEditableSelection() final;
    int getPasteboardItemsCount() final;
    RefPtr<WebCore::DocumentFragment> documentFragmentFromDelegate(int index) final;
    bool performsTwoStepPaste(WebCore::DocumentFragment*) final;
    void updateStringForFind(const String&) final { }
    bool shouldRevealCurrentSelectionAfterInsertion() const final;
    bool shouldSuppressPasswordEcho() const final;
#endif

    bool performTwoStepDrop(WebCore::DocumentFragment&, const WebCore::SimpleRange& destination, bool isMove) final;
    
    bool shouldEraseMarkersAfterChangeSelection(WebCore::TextCheckingType) const final;
    void ignoreWordInSpellDocument(const String&) final;
    void learnWord(const String&) final;
    void checkSpellingOfString(StringView, int* misspellingLocation, int* misspellingLength) final;
    void checkGrammarOfString(StringView, Vector<WebCore::GrammarDetail>&, int* badGrammarLocation, int* badGrammarLength) final;
    Vector<WebCore::TextCheckingResult> checkTextOfParagraph(StringView, OptionSet<WebCore::TextCheckingType> checkingTypes, const WebCore::VisibleSelection& currentSelection) final;
    void updateSpellingUIWithGrammarString(const String&, const WebCore::GrammarDetail&) final;
    void updateSpellingUIWithMisspelledWord(const String&) final;
    void showSpellingUI(bool show) final;
    bool spellingUIIsShowing() final;
    void getGuessesForWord(const String& word, const String& context, const WebCore::VisibleSelection& currentSelection, Vector<String>& guesses) final;

    void setInputMethodState(WebCore::Element*) final;
    void requestCheckingOfString(WebCore::TextCheckingRequest&, const WebCore::VisibleSelection& currentSelection) final;

#if PLATFORM(MAC)
    void requestCandidatesForSelection(const WebCore::VisibleSelection&) final;
    void handleRequestedCandidates(NSInteger, NSArray<NSTextCheckingResult *> *);
    void handleAcceptedCandidateWithSoftSpaces(WebCore::TextCheckingResult) final;
#endif

    void registerUndoOrRedoStep(WebCore::UndoStep&, bool isRedo);

#if PLATFORM(IOS_FAMILY)
    bool shouldAllowSingleClickToChangeSelection(WebCore::Node& targetNode, const WebCore::VisibleSelection& newSelection) const;
#endif

    WebView *m_webView;
    RetainPtr<WebEditorUndoTarget> m_undoTarget;
    bool m_haveUndoRedoOperations { false };
    
    HashMap<WebCore::TextCheckingRequestIdentifier, Ref<WebCore::TextCheckingRequest>> m_requestsInFlight;

#if PLATFORM(IOS_FAMILY)
    bool m_delayingContentChangeNotifications { false };
    bool m_hasDelayedContentChangeNotification { false };
#endif

    WebCore::VisibleSelection m_lastSelectionForRequestedCandidates;

#if PLATFORM(MAC)
    RetainPtr<NSString> m_paragraphContextForCandidateRequest;
    NSRange m_rangeForCandidates;
    NSInteger m_lastCandidateRequestSequenceNumber;
#endif

#if ENABLE(TEXT_CARET)
    bool m_lastSelectionWasPainted { false };
#endif

    enum class EditorStateIsContentEditable { No, Yes, Unset };
    EditorStateIsContentEditable m_lastEditorStateWasContentEditable { EditorStateIsContentEditable::Unset };
};

inline NSSelectionAffinity kit(WebCore::Affinity affinity)
{
    switch (affinity) {
    case WebCore::Affinity::Upstream:
        return NSSelectionAffinityUpstream;
    case WebCore::Affinity::Downstream:
        return NSSelectionAffinityDownstream;
    }
    ASSERT_NOT_REACHED();
    return NSSelectionAffinityDownstream;
}

inline WebCore::Affinity core(NSSelectionAffinity affinity)
{
    switch (affinity) {
    case NSSelectionAffinityUpstream:
        return WebCore::Affinity::Upstream;
    case NSSelectionAffinityDownstream:
        return WebCore::Affinity::Downstream;
    }
    ASSERT_NOT_REACHED();
    return WebCore::Affinity::Downstream;
}

#if PLATFORM(IOS_FAMILY)

inline bool WebEditorClient::isGrammarCheckingEnabled()
{
    return false;
}

inline void WebEditorClient::toggleGrammarChecking()
{
}

inline int WebEditorClient::spellCheckerDocumentTag()
{
    return 0;
}

inline bool WebEditorClient::shouldEraseMarkersAfterChangeSelection(WebCore::TextCheckingType) const
{
    return true;
}

inline void WebEditorClient::ignoreWordInSpellDocument(const String&)
{
}

inline void WebEditorClient::learnWord(const String&)
{
}

inline void WebEditorClient::checkSpellingOfString(StringView, int* misspellingLocation, int* misspellingLength)
{
}

inline void WebEditorClient::checkGrammarOfString(StringView, Vector<WebCore::GrammarDetail>&, int* badGrammarLocation, int* badGrammarLength)
{
}

inline void WebEditorClient::updateSpellingUIWithGrammarString(const String&, const WebCore::GrammarDetail&)
{
}

inline void WebEditorClient::updateSpellingUIWithMisspelledWord(const String&)
{
}

inline void WebEditorClient::showSpellingUI(bool show)
{
}

inline bool WebEditorClient::spellingUIIsShowing()
{
    return false;
}

inline void WebEditorClient::getGuessesForWord(const String&, const String&, const WebCore::VisibleSelection&, Vector<String>&)
{
}

#endif
