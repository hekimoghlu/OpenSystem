/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 8, 2024.
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

#import <WebCore/WritingToolsTypes.h>
#import <pal/spi/cocoa/WritingToolsSPI.h>
#import <wtf/RetainPtr.h>

namespace WebCore {

namespace WritingTools {
enum class Behavior : uint8_t;
enum class EditAction : uint8_t;
enum class ReplacementBehavior : uint8_t;
enum class ReplacementState : uint8_t;
enum class RequestedTool : uint16_t;
enum class SessionCompositionType : uint8_t;
enum class SessionType : uint8_t;
enum class TextSuggestionState : uint8_t;

struct Context;
struct Replacement;
struct Session;
}

}

namespace WebKit {

#pragma mark - Conversions from web types to platform types.

PlatformWritingToolsBehavior convertToPlatformWritingToolsBehavior(WebCore::WritingTools::Behavior);

WTRequestedTool convertToPlatformRequestedTool(WebCore::WritingTools::RequestedTool);

WTTextSuggestionState convertToPlatformTextSuggestionState(WebCore::WritingTools::TextSuggestionState);

RetainPtr<WTContext> convertToPlatformContext(const WebCore::WritingTools::Context&);

#pragma mark - Conversions from platform types to web types.

WebCore::WritingTools::Behavior convertToWebWritingToolsBehavior(PlatformWritingToolsBehavior);

WebCore::WritingTools::RequestedTool convertToWebRequestedTool(WTRequestedTool);

WebCore::WritingTools::TextSuggestionState convertToWebTextSuggestionState(WTTextSuggestionState);

WebCore::WritingTools::Action convertToWebAction(WTAction);

WebCore::WritingTools::SessionType convertToWebSessionType(WTSessionType);

WebCore::WritingTools::SessionCompositionType convertToWebCompositionSessionType(WTCompositionSessionType);

std::optional<WebCore::WritingTools::Context> convertToWebContext(WTContext *);

std::optional<WebCore::WritingTools::Session> convertToWebSession(WTSession *);

std::optional<WebCore::WritingTools::TextSuggestion> convertToWebTextSuggestion(WTTextSuggestion *);

} // namespace WebKit

#endif
