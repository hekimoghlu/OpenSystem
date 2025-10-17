/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 24, 2023.
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
#include "config.h"
#include "EditAction.h"

#include "LocalizedStrings.h"
#include <wtf/text/WTFString.h>

namespace WebCore {

String undoRedoLabel(EditAction editAction)
{
    switch (editAction) {
    case EditAction::Unspecified:
    case EditAction::Insert:
    case EditAction::InsertReplacement:
    case EditAction::InsertFromDrop:
        return { };
    case EditAction::SetColor:
        return WEB_UI_STRING_KEY("Set Color", "Set Color (Undo action name)", "Undo action name");
    case EditAction::SetBackgroundColor:
        return WEB_UI_STRING_KEY("Set Background Color", "Set Background Color (Undo action name)", "Undo action name");
    case EditAction::TurnOffKerning:
        return WEB_UI_STRING_KEY("Turn Off Kerning", "Turn Off Kerning (Undo action name)", "Undo action name");
    case EditAction::TightenKerning:
        return WEB_UI_STRING_KEY("Tighten Kerning", "Tighten Kerning (Undo action name)", "Undo action name");
    case EditAction::LoosenKerning:
        return WEB_UI_STRING_KEY("Loosen Kerning", "Loosen Kerning (Undo action name)", "Undo action name");
    case EditAction::UseStandardKerning:
        return WEB_UI_STRING_KEY("Use Standard Kerning", "Use Standard Kerning (Undo action name)", "Undo action name");
    case EditAction::TurnOffLigatures:
        return WEB_UI_STRING_KEY("Turn Off Ligatures", "Turn Off Ligatures (Undo action name)", "Undo action name");
    case EditAction::UseStandardLigatures:
        return WEB_UI_STRING_KEY("Use Standard Ligatures", "Use Standard Ligatures (Undo action name)", "Undo action name");
    case EditAction::UseAllLigatures:
        return WEB_UI_STRING_KEY("Use All Ligatures", "Use All Ligatures (Undo action name)", "Undo action name");
    case EditAction::RaiseBaseline:
        return WEB_UI_STRING_KEY("Raise Baseline", "Raise Baseline (Undo action name)", "Undo action name");
    case EditAction::LowerBaseline:
        return WEB_UI_STRING_KEY("Lower Baseline", "Lower Baseline (Undo action name)", "Undo action name");
    case EditAction::SetTraditionalCharacterShape:
        return WEB_UI_STRING_KEY("Set Traditional Character Shape", "Set Traditional Character Shape (Undo action name)", "Undo action name");
    case EditAction::SetFont:
        return WEB_UI_STRING_KEY("Set Font", "Set Font (Undo action name)", "Undo action name");
    case EditAction::ChangeAttributes:
        return WEB_UI_STRING_KEY("Change Attributes", "Change Attributes (Undo action name)", "Undo action name");
    case EditAction::AlignLeft:
        return WEB_UI_STRING_KEY("Align Left", "Align Left (Undo action name)", "Undo action name");
    case EditAction::AlignRight:
        return WEB_UI_STRING_KEY("Align Right", "Align Right (Undo action name)", "Undo action name");
    case EditAction::Center:
        return WEB_UI_STRING_KEY("Center", "Center (Undo action name)", "Undo action name");
    case EditAction::Justify:
        return WEB_UI_STRING_KEY("Justify", "Justify (Undo action name)", "Undo action name");
    case EditAction::SetInlineWritingDirection:
    case EditAction::SetBlockWritingDirection:
        return WEB_UI_STRING_KEY("Set Writing Direction", "Set Writing Direction (Undo action name)", "Undo action name");
    case EditAction::Subscript:
        return WEB_UI_STRING_KEY("Subscript", "Subscript (Undo action name)", "Undo action name");
    case EditAction::Superscript:
        return WEB_UI_STRING_KEY("Superscript", "Superscript (Undo action name)", "Undo action name");
    case EditAction::Underline:
        return WEB_UI_STRING_KEY("Underline", "Underline (Undo action name)", "Undo action name");
    case EditAction::StrikeThrough:
        return WEB_UI_STRING_KEY("StrikeThrough", "StrikeThrough (Undo action name)", "Undo action name");
    case EditAction::Outline:
        return WEB_UI_STRING_KEY("Outline", "Outline (Undo action name)", "Undo action name");
    case EditAction::Unscript:
        return WEB_UI_STRING_KEY("Unscript", "Unscript (Undo action name)", "Undo action name");
    case EditAction::DeleteByDrag:
        return WEB_UI_STRING_KEY("Drag", "Drag (Undo action name)", "Undo action name");
    case EditAction::Cut:
        return WEB_UI_STRING_KEY("Cut", "Cut (Undo action name)", "Undo action name");
    case EditAction::Bold:
        return WEB_UI_STRING_KEY("Bold", "Bold (Undo action name)", "Undo action name");
    case EditAction::Italics:
        return WEB_UI_STRING_KEY("Italics", "Italics (Undo action name)", "Undo action name");
    case EditAction::Delete:
        return WEB_UI_STRING_KEY("Delete", "Delete (Undo action name)", "Undo action name");
    case EditAction::Dictation:
        return WEB_UI_STRING_KEY("Dictation", "Dictation (Undo action name)", "Undo action name");
    case EditAction::Paste:
        return WEB_UI_STRING_KEY("Paste", "Paste (Undo action name)", "Undo action name");
    case EditAction::PasteFont:
        return WEB_UI_STRING_KEY("Paste Font", "Paste Font (Undo action name)", "Undo action name");
    case EditAction::PasteRuler:
        return WEB_UI_STRING_KEY("Paste Ruler", "Paste Ruler (Undo action name)", "Undo action name");
    case EditAction::TypingDeleteSelection:
    case EditAction::TypingDeleteBackward:
    case EditAction::TypingDeleteForward:
    case EditAction::TypingDeleteWordBackward:
    case EditAction::TypingDeleteWordForward:
    case EditAction::TypingDeleteLineBackward:
    case EditAction::TypingDeleteLineForward:
    case EditAction::TypingDeletePendingComposition:
    case EditAction::TypingDeleteFinalComposition:
    case EditAction::TypingInsertText:
    case EditAction::TypingInsertLineBreak:
    case EditAction::TypingInsertParagraph:
    case EditAction::TypingInsertPendingComposition:
    case EditAction::TypingInsertFinalComposition:
        return WEB_UI_STRING_KEY("Typing", "Typing (Undo action name)", "Undo action name");
    case EditAction::CreateLink:
        return WEB_UI_STRING_KEY("Create Link", "Create Link (Undo action name)", "Undo action name");
    case EditAction::Unlink:
        return WEB_UI_STRING_KEY("Unlink", "Unlink (Undo action name)", "Undo action name");
    case EditAction::InsertUnorderedList:
    case EditAction::InsertOrderedList:
        return WEB_UI_STRING_KEY("Insert List", "Insert List (Undo action name)", "Undo action name");
    case EditAction::FormatBlock:
        return WEB_UI_STRING_KEY("Formatting", "Format Block (Undo action name)", "Undo action name");
    case EditAction::Indent:
        return WEB_UI_STRING_KEY("Indent", "Indent (Undo action name)", "Undo action name");
    case EditAction::Outdent:
        return WEB_UI_STRING_KEY("Outdent", "Outdent (Undo action name)", "Undo action name");
    // FIXME: We should give internal clients a way to override these undo names. For instance, Mail refers to ordered and unordered lists as "numbered" and "bulleted" lists, respectively,
    // despite the fact that ordered and unordered lists are not necessarily displayed using bullets and numerals.
    case EditAction::ConvertToOrderedList:
        return WEB_UI_STRING_KEY("Convert to Ordered List", "Convert to Ordered List (Undo action name)", "Undo action name");
    case EditAction::ConvertToUnorderedList:
        return WEB_UI_STRING_KEY("Convert to Unordered List", "Convert to Unordered List (Undo action name)", "Undo action name");
    case EditAction::RemoveBackground:
        return WEB_UI_STRING_KEY("Remove Background", "Remove Background (Undo action name)", "Undo action name");
    }
    return { };
}

} // namespace WebCore
