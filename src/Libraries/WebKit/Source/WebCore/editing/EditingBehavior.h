/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 8, 2022.
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

#include "EditingBehaviorType.h"

namespace WebCore {

class EditingBehavior {
public:
    explicit EditingBehavior(EditingBehaviorType type)
        : m_type(type)
    {
    }

    // Individual functions for each case where we have more than one style of editing behavior.
    // Create a new function for any platform difference so we can control it here.

    // When extending a selection beyond the top or bottom boundary of an editable area,
    // maintain the horizontal position on Windows but extend it to the boundary of the editable
    // content on Mac.
    bool shouldMoveCaretToHorizontalBoundaryWhenPastTopOrBottom() const
    {
        return m_type != EditingBehaviorType::Windows;
    }

    // On Windows, selections should always be considered as directional, regardless if it is
    // mouse-based or keyboard-based.
    bool shouldConsiderSelectionAsDirectional() const { return m_type != EditingBehaviorType::Mac && m_type != EditingBehaviorType::iOS; }

    // On Mac, when revealing a selection (for example as a result of a Find operation on the Browser),
    // content should be scrolled such that the selection gets certer aligned.
    bool shouldCenterAlignWhenSelectionIsRevealed() const { return m_type == EditingBehaviorType::Mac || m_type == EditingBehaviorType::iOS; }

    // On Mac, style is considered present when present at the beginning of selection. On other platforms,
    // style has to be present throughout the selection.
    bool shouldToggleStyleBasedOnStartOfSelection() const { return m_type == EditingBehaviorType::Mac || m_type == EditingBehaviorType::iOS; }

    // Standard Mac behavior when extending to a boundary is grow the selection rather than leaving the base
    // in place and moving the extent. Matches NSTextView.
    bool shouldAlwaysGrowSelectionWhenExtendingToBoundary() const { return m_type == EditingBehaviorType::Mac || m_type == EditingBehaviorType::iOS; }

    // On Mac, when processing a contextual click, the object being clicked upon should be selected.
    bool shouldSelectOnContextualMenuClick() const { return m_type == EditingBehaviorType::Mac; }

    // On Linux, should be able to get and insert spelling suggestions without selecting the misspelled word.
    bool shouldAllowSpellingSuggestionsWithoutSelection() const
    {
        return m_type == EditingBehaviorType::Unix;
    }
    
    // On Mac and Windows, pressing backspace (when it isn't handled otherwise) should navigate back.
    bool shouldNavigateBackOnBackspace() const
    {
        return m_type != EditingBehaviorType::Unix;
    }

    // On Mac, selecting backwards by word/line from the middle of a word/line, and then going
    // forward leaves the caret back in the middle with no selection, instead of directly selecting
    // to the other end of the line/word (Unix/Windows behavior).
    bool shouldExtendSelectionByWordOrLineAcrossCaret() const { return m_type != EditingBehaviorType::Mac && m_type != EditingBehaviorType::iOS; }

    // Based on native behavior, when using ctrl(alt)+arrow to move caret by word, ctrl(alt)+left arrow moves caret to
    // immediately before the word in all platforms, for example, the word break positions are: "|abc |def |hij |opq".
    // But ctrl+right arrow moves caret to "abc |def |hij |opq" on Windows and "abc| def| hij| opq|" on Mac and Linux.
    bool shouldSkipSpaceWhenMovingRight() const { return m_type == EditingBehaviorType::Windows; }

    // On iOS the last entered character in a secure filed is shown momentarily, removing and adding back the
    // space when deleting password cause space been showed insecurely.
    bool shouldRebalanceWhiteSpacesInSecureField() const { return m_type != EditingBehaviorType::iOS; }

    bool shouldSelectBasedOnDictionaryLookup() const { return m_type == EditingBehaviorType::Mac || m_type == EditingBehaviorType::iOS; }

    // Linux and Windows always extend selections from the extent endpoint.
    bool shouldAlwaysExtendSelectionFromExtentEndpoint() const { return m_type != EditingBehaviorType::Mac && m_type != EditingBehaviorType::iOS; }

    // On iOS, we don't want to select all the text when focusing a field. Instead, match platform behavior by going to the end of the line.
    bool shouldMoveSelectionToEndWhenFocusingTextInput() const { return m_type == EditingBehaviorType::iOS; }
    
    // On iOS, when smart delete is on, it is always on, and should do not additional checks (i.e. TextGranularity::WordGranularity).
    bool shouldAlwaysSmartDelete() const { return m_type == EditingBehaviorType::iOS; }
    
    // On iOS, we should turn on smart insert and delete and newlines around paragraphs to match UIKit behaviour.
    bool shouldSmartInsertDeleteParagraphs() const { return m_type == EditingBehaviorType::iOS; }

private:
    EditingBehaviorType m_type;
};

} // namespace WebCore
