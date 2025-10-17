/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 29, 2023.
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
#if ENABLE(WRITING_TOOLS)

#import "config.h"
#import "PlatformWritingToolsUtilities.h"

#import <SoftLinking/WeakLinking.h>
#import <WebCore/WritingToolsTypes.h>

WEAK_IMPORT_OBJC_CLASS(WTContext);

namespace WebKit {

#pragma mark - Conversions from web types to platform types.

PlatformWritingToolsBehavior convertToPlatformWritingToolsBehavior(WebCore::WritingTools::Behavior behavior)
{
    switch (behavior) {
    case WebCore::WritingTools::Behavior::None:
        return PlatformWritingToolsBehaviorNone;

    case WebCore::WritingTools::Behavior::Default:
        return PlatformWritingToolsBehaviorDefault;

    case WebCore::WritingTools::Behavior::Limited:
        return PlatformWritingToolsBehaviorLimited;

    case WebCore::WritingTools::Behavior::Complete:
        return PlatformWritingToolsBehaviorComplete;
    }
}

WTRequestedTool convertToPlatformRequestedTool(WebCore::WritingTools::RequestedTool tool)
{
    return static_cast<WTRequestedTool>(tool);
}

WTTextSuggestionState convertToPlatformTextSuggestionState(WebCore::WritingTools::TextSuggestion::State state)
{
    switch (state) {
    case WebCore::WritingTools::TextSuggestion::State::Pending:
        return WTTextSuggestionStatePending;
    case WebCore::WritingTools::TextSuggestion::State::Reviewing:
        return WTTextSuggestionStateReviewing;
    case WebCore::WritingTools::TextSuggestion::State::Rejected:
        return WTTextSuggestionStateRejected;
    case WebCore::WritingTools::TextSuggestion::State::Invalid:
        return WTTextSuggestionStateInvalid;
    }
}

RetainPtr<WTContext> convertToPlatformContext(const WebCore::WritingTools::Context& contextData)
{
    return adoptNS([[WTContext alloc] initWithAttributedText:contextData.attributedText.nsAttributedString().get() range:NSMakeRange(contextData.range.location, contextData.range.length)]);
}

#pragma mark - Conversions from platform types to web types.

WebCore::WritingTools::Behavior convertToWebWritingToolsBehavior(PlatformWritingToolsBehavior behavior)
{
    switch (behavior) {
    case PlatformWritingToolsBehaviorNone:
        return WebCore::WritingTools::Behavior::None;

    case PlatformWritingToolsBehaviorDefault:
        return WebCore::WritingTools::Behavior::Default;

    case PlatformWritingToolsBehaviorLimited:
        return WebCore::WritingTools::Behavior::Limited;

    case PlatformWritingToolsBehaviorComplete:
        return WebCore::WritingTools::Behavior::Complete;
    }
}

WebCore::WritingTools::RequestedTool convertToWebRequestedTool(WTRequestedTool tool)
{
    return static_cast<WebCore::WritingTools::RequestedTool>(tool);
}

WebCore::WritingTools::TextSuggestion::State convertToWebTextSuggestionState(WTTextSuggestionState state)
{
    switch (state) {
    case WTTextSuggestionStatePending:
        return WebCore::WritingTools::TextSuggestion::State::Pending;
    case WTTextSuggestionStateReviewing:
        return WebCore::WritingTools::TextSuggestion::State::Reviewing;
    case WTTextSuggestionStateRejected:
        return WebCore::WritingTools::TextSuggestion::State::Rejected;
    case WTTextSuggestionStateInvalid:
        return WebCore::WritingTools::TextSuggestion::State::Invalid;

    // FIXME: Remove this default case once the WTTextSuggestionStateAccepted case is no longer in the build.
    default:
        ASSERT_NOT_REACHED();
        return WebCore::WritingTools::TextSuggestion::State::Invalid;
    }
}

WebCore::WritingTools::Action convertToWebAction(WTAction action)
{
    switch (action) {
    case WTActionShowOriginal:
        return WebCore::WritingTools::Action::ShowOriginal;
    case WTActionShowRewritten:
        return WebCore::WritingTools::Action::ShowRewritten;
    case WTActionCompositionRestart:
        return WebCore::WritingTools::Action::Restart;

    // FIXME (rdar://141822568): Handle all Writing Tools actions.
    default:
        return WebCore::WritingTools::Action::Restart;
    }
}

WebCore::WritingTools::Session::Type convertToWebSessionType(WTSessionType type)
{
    switch (type) {
    case WTSessionTypeProofreading:
        return WebCore::WritingTools::Session::Type::Proofreading;
    case WTSessionTypeComposition:
        return WebCore::WritingTools::Session::Type::Composition;
    }
}

WebCore::WritingTools::Session::CompositionType convertToWebCompositionSessionType(WTCompositionSessionType type)
{
    switch (type) {
    case WTCompositionSessionTypeNone:
        return WebCore::WritingTools::Session::CompositionType::None;

    case WTCompositionSessionTypeCompose:
        return WebCore::WritingTools::Session::CompositionType::Compose;

    case WTCompositionSessionTypeSmartReply:
        return WebCore::WritingTools::Session::CompositionType::SmartReply;

    default:
        return WebCore::WritingTools::Session::CompositionType::Other;
    }
}

std::optional<WebCore::WritingTools::Context> convertToWebContext(WTContext *context)
{
    auto contextUUID = WTF::UUID::fromNSUUID(context.uuid);
    if (!contextUUID)
        return std::nullopt;

    return { { *contextUUID, WebCore::AttributedString::fromNSAttributedString(context.attributedText), { context.range } } };
}

std::optional<WebCore::WritingTools::Session> convertToWebSession(WTSession *session)
{
    auto sessionUUID = WTF::UUID::fromNSUUID(session.uuid);
    if (!sessionUUID)
        return std::nullopt;

    return { { *sessionUUID, convertToWebSessionType(session.type), convertToWebCompositionSessionType(session.compositionSessionType) } };
}

std::optional<WebCore::WritingTools::TextSuggestion> convertToWebTextSuggestion(WTTextSuggestion *suggestion)
{
    auto suggestionUUID = WTF::UUID::fromNSUUID(suggestion.uuid);
    if (!suggestionUUID)
        return std::nullopt;

    return { { *suggestionUUID, { suggestion.originalRange }, { suggestion.replacement }, convertToWebTextSuggestionState(suggestion.state) } };
}

} // namespace WebKit

#endif
