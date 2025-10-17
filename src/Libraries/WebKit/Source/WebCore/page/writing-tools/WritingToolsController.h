/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 15, 2023.
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

#if ENABLE(WRITING_TOOLS)

#import "Range.h"
#import "WritingToolsCompositionCommand.h"
#import "WritingToolsTypes.h"
#import <wtf/CheckedPtr.h>
#import <wtf/TZoneMalloc.h>
#import <wtf/WeakPtr.h>

namespace WebCore {

class CompositeEditCommand;
class EditCommandComposition;
class Document;
class DocumentFragment;
class DocumentMarker;
class Node;
class Page;

struct SimpleRange;

enum class TextAnimationRunMode : uint8_t;

class WritingToolsController final : public CanMakeWeakPtr<WritingToolsController>, public CanMakeCheckedPtr<WritingToolsController> {
    WTF_MAKE_TZONE_ALLOCATED(WritingToolsController);
    WTF_MAKE_NONCOPYABLE(WritingToolsController);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(WritingToolsController);

public:
    explicit WritingToolsController(Page&);

    void willBeginWritingToolsSession(const std::optional<WritingTools::Session>&, CompletionHandler<void(const Vector<WritingTools::Context>&)>&&);

    void didBeginWritingToolsSession(const WritingTools::Session&, const Vector<WritingTools::Context>&);

    void proofreadingSessionDidReceiveSuggestions(const WritingTools::Session&, const Vector<WritingTools::TextSuggestion>&, const CharacterRange&, const WritingTools::Context&, bool finished);

    void proofreadingSessionDidUpdateStateForSuggestion(const WritingTools::Session&, WritingTools::TextSuggestion::State, const WritingTools::TextSuggestion&, const WritingTools::Context&);

    void willEndWritingToolsSession(const WritingTools::Session&, bool accepted);

    void didEndWritingToolsSession(const WritingTools::Session&, bool accepted);

    void compositionSessionDidReceiveTextWithReplacementRange(const WritingTools::Session&, const AttributedString&, const CharacterRange&, const WritingTools::Context&, bool finished);

    void writingToolsSessionDidReceiveAction(const WritingTools::Session&, WritingTools::Action);

    void updateStateForSelectedSuggestionIfNeeded();

    void respondToUnappliedEditing(EditCommandComposition*);
    void respondToReappliedEditing(EditCommandComposition*);

    // FIXME: Refactor `TextAnimationController` in such a way so as to not explicitly depend on `WritingToolsController`,
    // and then remove these methods after doing so.
    std::optional<SimpleRange> activeSessionRange() const;
    void intelligenceTextAnimationsDidComplete();

private:
    struct CompositionState : CanMakeCheckedPtr<CompositionState> {
        WTF_MAKE_STRUCT_FAST_ALLOCATED;
        WTF_STRUCT_OVERRIDE_DELETE_FOR_CHECKED_PTR(CompositionState);

        enum class ClearStateDeferralReason : uint8_t {
            SessionInProgress = 1 << 0,
            AnimationInProgress = 1 << 1,
        };

        CompositionState(const Vector<Ref<WritingToolsCompositionCommand>>& unappliedCommands, const Vector<Ref<WritingToolsCompositionCommand>>& reappliedCommands, const WritingTools::Session& session)
            : unappliedCommands(unappliedCommands)
            , reappliedCommands(reappliedCommands)
            , session(session)
        {
        }

        // These two vectors should never have the same command in both of them.
        Vector<Ref<WritingToolsCompositionCommand>> unappliedCommands;
        Vector<Ref<WritingToolsCompositionCommand>> reappliedCommands;
        WritingTools::Session session;
        OptionSet<ClearStateDeferralReason> clearStateDeferralReasons;

        bool shouldCommitAfterReplacement { false };
        std::optional<CharacterRange> replacedRange;
        std::optional<CharacterRange> pendingReplacedRange;
    };

    struct ProofreadingState : CanMakeCheckedPtr<ProofreadingState> {
        WTF_MAKE_STRUCT_FAST_ALLOCATED;
        WTF_STRUCT_OVERRIDE_DELETE_FOR_CHECKED_PTR(ProofreadingState);

        ProofreadingState(const Ref<Range>& contextRange, const WritingTools::Session& session, int replacementLocationOffset)
            : contextRange(contextRange)
            , session(session)
            , replacementLocationOffset(replacementLocationOffset)
        {
        }

        Ref<Range> contextRange;
        WritingTools::Session session;
        int replacementLocationOffset { 0 };
    };

    template<WritingTools::Session::Type Type>
    struct StateFromSessionType { };

    template<>
    struct StateFromSessionType<WritingTools::Session::Type::Proofreading> {
        using Value = ProofreadingState;
    };

    template<>
    struct StateFromSessionType<WritingTools::Session::Type::Composition> {
        using Value = CompositionState;
    };

    class EditingScope {
        WTF_MAKE_TZONE_ALLOCATED(EditingScope);
        WTF_MAKE_NONCOPYABLE(EditingScope);
    public:
        EditingScope(Document&);
        ~EditingScope();

    private:
        RefPtr<Document> m_document;
        bool m_editingWasSuppressed;
    };

    static CharacterRange characterRange(const SimpleRange& scope, const SimpleRange&);
    static SimpleRange resolveCharacterRange(const SimpleRange& scope, CharacterRange);
    static uint64_t characterCount(const SimpleRange&);
    static String plainText(const SimpleRange&);

    template<WritingTools::Session::Type Type>
    StateFromSessionType<Type>::Value* currentState();

    template<WritingTools::Session::Type Type>
    const StateFromSessionType<Type>::Value* currentState() const;

    std::optional<std::tuple<Node&, DocumentMarker&>> findTextSuggestionMarkerContainingRange(const SimpleRange&) const;
    std::optional<std::tuple<Node&, DocumentMarker&>> findTextSuggestionMarkerByID(const SimpleRange& outerRange, const WritingTools::TextSuggestion::ID&) const;

    void replaceContentsOfRangeInSession(ProofreadingState&, const SimpleRange&, const String&);
    void replaceContentsOfRangeInSession(CompositionState&, const SimpleRange&, const AttributedString&, WritingToolsCompositionCommand::State);

    void compositionSessionDidFinishReplacement();
    void compositionSessionDidFinishReplacement(const WTF::UUID& sourceAnimationUUID, const WTF::UUID& destinationAnimationUUID, const CharacterRange&, const String&);

    void smartReplySessionDidReceiveTextWithReplacementRange(const WritingTools::Session&, const AttributedString&, const CharacterRange&, const WritingTools::Context&, bool finished);

    void compositionSessionDidReceiveTextWithReplacementRangeAsync(const WTF::UUID&, const WTF::UUID&, const AttributedString&, const CharacterRange&, const WritingTools::Context&, bool finished, TextAnimationRunMode);

    void showOriginalCompositionForSession();
    void showRewrittenCompositionForSession();
    void restartCompositionForSession();

    template<CompositionState::ClearStateDeferralReason Reason>
    void removeCompositionClearStateDeferralReason();

    template<WritingTools::Session::Type Type>
    void writingToolsSessionDidReceiveAction(WritingTools::Action);

    template<WritingTools::Session::Type Type>
    void willEndWritingToolsSession(bool accepted);

    template<WritingTools::Session::Type Type>
    void didEndWritingToolsSession(bool accepted);

    void commitComposition(CompositionState&, Document&);

    RefPtr<Document> document() const;

    WeakPtr<Page> m_page;

    std::variant<std::monostate, ProofreadingState, CompositionState> m_state;
};

} // namespace WebKit

#endif
