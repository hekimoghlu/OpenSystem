/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 11, 2023.
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

namespace WTF {
class String;
}

namespace WebCore {

enum class EditAction : uint8_t {
    AlignLeft,
    AlignRight,
    Bold,
    Center,
    ChangeAttributes,
    ConvertToOrderedList,
    ConvertToUnorderedList,
    CreateLink,
    Cut,
    Delete,
    DeleteByDrag,
    Dictation,
    FormatBlock,
    Indent,
    Insert,
    InsertFromDrop,
    InsertOrderedList,
    InsertReplacement,
    InsertUnorderedList,
    Italics,
    Justify,
    LoosenKerning,
    LowerBaseline,
    Outdent,
    Outline,
    Paste,
    PasteFont,
    PasteRuler,
    RaiseBaseline,
    RemoveBackground,
    SetBackgroundColor,
    SetBlockWritingDirection,
    SetColor,
    SetFont,
    SetInlineWritingDirection,
    SetTraditionalCharacterShape,
    StrikeThrough,
    Subscript,
    Superscript,
    TightenKerning,
    TurnOffKerning,
    TurnOffLigatures,
    TypingDeleteBackward,
    TypingDeleteFinalComposition,
    TypingDeleteForward,
    TypingDeleteLineBackward,
    TypingDeleteLineForward,
    TypingDeletePendingComposition,
    TypingDeleteSelection,
    TypingDeleteWordBackward,
    TypingDeleteWordForward,
    TypingInsertFinalComposition,
    TypingInsertLineBreak,
    TypingInsertParagraph,
    TypingInsertPendingComposition,
    TypingInsertText,
    Underline,
    Unlink,
    Unscript,
    Unspecified,
    UseAllLigatures,
    UseStandardKerning,
    UseStandardLigatures,
};

WTF::String undoRedoLabel(EditAction);

} // namespace WebCore
