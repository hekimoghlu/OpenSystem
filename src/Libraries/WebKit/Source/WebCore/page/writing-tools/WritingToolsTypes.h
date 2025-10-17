/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 26, 2023.
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

#include "AttributedString.h"
#include "CharacterRange.h"
#include <wtf/Forward.h>

namespace WebCore {
namespace WritingTools {

enum class Behavior : uint8_t {
    None,
    Default,
    Limited,
    Complete,
};

enum class Action : uint8_t {
    ShowOriginal,
    ShowRewritten,
    Restart,
};

enum class RequestedTool : uint16_t {
    // Opaque type to transitively convert to/from WTRequestedTool.
};

#pragma mark - Session

enum class SessionType : uint8_t {
    Proofreading,
    Composition,
};

enum class SessionCompositionType : uint8_t {
    None,
    Compose,
    SmartReply,
    Other,
};

using SessionID = WTF::UUID;

struct Session {
    using ID = SessionID;

    using Type = SessionType;
    using CompositionType = SessionCompositionType;

    ID identifier;
    Type type { Type::Composition };
    CompositionType compositionType { CompositionType::None };
};

#pragma mark - Context

using ContextID = WTF::UUID;

struct Context {
    using ID = ContextID;

    ID identifier;
    AttributedString attributedText;
    CharacterRange range;
};

#pragma mark - TextSuggestion

enum class TextSuggestionState : uint8_t {
    Pending,
    Reviewing,
    Rejected,
    Invalid,
};

using TextSuggestionID = WTF::UUID;

struct TextSuggestion {
    using ID = TextSuggestionID;

    using State = TextSuggestionState;

    ID identifier;
    CharacterRange originalRange;
    String replacement;
    State state;
};

} // namespace WritingTools
} // namespace WebCore

#endif
