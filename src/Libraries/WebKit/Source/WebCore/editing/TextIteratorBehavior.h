/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 30, 2024.
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

#include <wtf/OptionSet.h>

namespace WebCore {

enum class TextIteratorBehavior : uint16_t {
    // Used by selection preservation code. There should be one character emitted between every VisiblePosition
    // in the Range used to create the TextIterator.
    // FIXME <rdar://problem/6028818>: This functionality should eventually be phased out when we rewrite
    // moveParagraphs to not clone/destroy moved content.
    EmitsCharactersBetweenAllVisiblePositions = 1 << 0,

    EntersTextControls = 1 << 1,

    // Used when we want text for copying, pasting, and transposing.
    EmitsTextsWithoutTranscoding = 1 << 2,

    // Used when the visibility of the style should not affect text gathering.
    IgnoresStyleVisibility = 1 << 3,

    // Used when emitting the special 0xFFFC character is required. Children for replaced objects will be ignored.
    EmitsObjectReplacementCharacters = 1 << 4,

    // Used when pasting inside password field.
    EmitsOriginalText = 1 << 5,

    EmitsImageAltText = 1 << 6,

    BehavesAsIfNodesFollowing = 1 << 7,

    // Makes visiblity test take into account the visibility of the frame.
    // FIXME: This should probably be always on unless TextIteratorBehavior::IgnoresStyleVisibility is set.
    ClipsToFrameAncestors = 1 << 8,

    TraversesFlatTree = 1 << 9,

    EntersImageOverlays = 1 << 10,

    IgnoresUserSelectNone = 1 << 11,

    EmitsObjectReplacementCharactersForImages = 1 << 12,

#if ENABLE(ATTACHMENT_ELEMENT)
    EmitsObjectReplacementCharactersForAttachments = 1 << 13,
#endif

    // Used by accessibility to expose untransformed kana text.
    IgnoresFullSizeKana = 1 << 14
};

using TextIteratorBehaviors = OptionSet<TextIteratorBehavior>;

} // namespace WebCore
