/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 22, 2024.
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

#include "AlternativeTextClient.h"
#include "DocumentMarker.h"
#include "EventLoop.h"
#include "Position.h"
#include <variant>
#include <wtf/Noncopyable.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakRef.h>

namespace WebCore {
class AlternativeTextController;
}

namespace WTF {
template<typename T> struct IsDeprecatedWeakRefSmartPointerException;
template<> struct IsDeprecatedWeakRefSmartPointerException<WebCore::AlternativeTextController> : std::true_type { };
}

namespace WebCore {

class CompositeEditCommand;
class Document;
class EditCommand;
class EditCommandComposition;
class EditorClient;
class Event;
class TextCheckerClient;
class VisibleSelection;

struct DictationAlternative;
struct SimpleRange;
struct TextCheckingResult;

#if USE(AUTOCORRECTION_PANEL)
// These backslashes are for making style checker happy.
#define UNLESS_ENABLED(functionBody) \
;\

#else
#define UNLESS_ENABLED(functionBody) functionBody
#endif

class AlternativeTextController : public CanMakeWeakPtr<AlternativeTextController> {
    WTF_MAKE_TZONE_ALLOCATED(AlternativeTextController);
    WTF_MAKE_NONCOPYABLE(AlternativeTextController);
public:
    explicit AlternativeTextController(Document& document) UNLESS_ENABLED(: m_document(document) { })
    ~AlternativeTextController() UNLESS_ENABLED({ })

    void startAlternativeTextUITimer(AlternativeTextType) UNLESS_ENABLED({ })
    void stopAlternativeTextUITimer() UNLESS_ENABLED({ })

    void dismiss(ReasonForDismissingAlternativeText) UNLESS_ENABLED({ })
    void show(const SimpleRange& rangeToReplace, const String& replacement) UNLESS_ENABLED({ UNUSED_PARAM(rangeToReplace); UNUSED_PARAM(replacement); })

    // Return true if correction was applied, false otherwise.
    bool applyAutocorrectionBeforeTypingIfAppropriate() UNLESS_ENABLED({ return false; })

    void respondToUnappliedSpellCorrection(const VisibleSelection&, const String& corrected, const String& correction) UNLESS_ENABLED({ UNUSED_PARAM(corrected); UNUSED_PARAM(correction); })
    void respondToAppliedEditing(CompositeEditCommand*);
    void respondToUnappliedEditing(EditCommandComposition*) UNLESS_ENABLED({ })
    void respondToChangedSelection(const VisibleSelection& oldSelection) UNLESS_ENABLED({ UNUSED_PARAM(oldSelection); })

    void stopPendingCorrection(const VisibleSelection& oldSelection) UNLESS_ENABLED({ UNUSED_PARAM(oldSelection); })
    void applyPendingCorrection(const VisibleSelection& selectionAfterTyping) UNLESS_ENABLED({ UNUSED_PARAM(selectionAfterTyping); })

    void handleAlternativeTextUIResult(const String& result) UNLESS_ENABLED({ UNUSED_PARAM(result); })
    void handleCancelOperation() UNLESS_ENABLED({ })

    bool hasPendingCorrection() const UNLESS_ENABLED({ return false; })
    bool isSpellingMarkerAllowed(const SimpleRange& misspellingRange) const UNLESS_ENABLED({ UNUSED_PARAM(misspellingRange); return true; })
    bool isAutomaticSpellingCorrectionEnabled() UNLESS_ENABLED({ return false; })
    bool canEnableAutomaticSpellingCorrection() const UNLESS_ENABLED({ return false; })
    bool shouldRemoveMarkersUponEditing();

    void recordAutocorrectionResponse(AutocorrectionResponse, const String& replacedString, const SimpleRange& replacementRange) UNLESS_ENABLED({ UNUSED_PARAM(replacedString); UNUSED_PARAM(replacementRange); })
    void markReversed(const SimpleRange& changedRange) UNLESS_ENABLED({ UNUSED_PARAM(changedRange); })
    void markCorrection(const SimpleRange& replacedRange, const String& replacedString) UNLESS_ENABLED({ UNUSED_PARAM(replacedRange); UNUSED_PARAM(replacedString); })

    // This function returns false if the replacement should not be carried out.
    bool processMarkersOnTextToBeReplacedByResult(const TextCheckingResult&, const SimpleRange& rangeToBeReplaced, const String& stringToBeReplaced) UNLESS_ENABLED({ UNUSED_PARAM(rangeToBeReplaced); UNUSED_PARAM(stringToBeReplaced); return true; });
    void deletedAutocorrectionAtPosition(const Position&, const String& originalString) UNLESS_ENABLED({ UNUSED_PARAM(originalString); })

    bool insertDictatedText(const String&, const Vector<DictationAlternative>&, Event*);
    void removeDictationAlternativesForMarker(const DocumentMarker&);
    Vector<String> dictationAlternativesForMarker(const DocumentMarker&);
    void applyDictationAlternative(const String& alternativeString);

private:
#if USE(AUTOCORRECTION_PANEL)
    using AutocorrectionReplacement = String;

    String dismissSoon(ReasonForDismissingAlternativeText);
    void timerFired();
    void recordSpellcheckerResponseForModifiedCorrection(const SimpleRange& rangeOfCorrection, const String& corrected, const String& correction);

    bool shouldStartTimerFor(const DocumentMarker&, int endOffset) const;
    bool respondToMarkerAtEndOfWord(const DocumentMarker&, const Position& endOfWordPosition);

    EditorClient* editorClient();
    
    TextCheckerClient* textChecker();
    FloatRect rootViewRectForRange(const SimpleRange&) const;
    void markPrecedingWhitespaceForDeletedAutocorrectionAfterCommand(EditCommand*);

    EventLoopTimerHandle m_timer;
    std::optional<SimpleRange> m_rangeWithAlternative;
    bool m_isActive { };
    bool m_isDismissedByEditing { };
    AlternativeTextType m_type;
    String m_originalText;
    std::variant<AutocorrectionReplacement, DictationContext> m_details;

    String m_originalStringForLastDeletedAutocorrection;
    Position m_positionForLastDeletedAutocorrection;
#endif
#if USE(DICTATION_ALTERNATIVES) || USE(AUTOCORRECTION_PANEL)
    String markerDescriptionForAppliedAlternativeText(AlternativeTextType, DocumentMarkerType);
    void applyAlternativeTextToRange(const SimpleRange&, const String&, AlternativeTextType, OptionSet<DocumentMarkerType>);
    AlternativeTextClient* alternativeTextClient();
#endif
    Ref<Document> protectedDocument() const { return m_document.get(); }

    void removeCorrectionIndicatorMarkers();

    WeakRef<Document, WeakPtrImplWithEventTargetData> m_document;
};

#undef UNLESS_ENABLED

inline bool AlternativeTextController::shouldRemoveMarkersUponEditing()
{
#if USE(MARKER_REMOVAL_UPON_EDITING)
    return true;
#else
    return false;
#endif
}

} // namespace WebCore
