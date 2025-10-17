/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 23, 2024.
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

namespace WebCore {
    
enum AXTextStateChangeType {
    AXTextStateChangeTypeUnknown,
    AXTextStateChangeTypeEdit,
    AXTextStateChangeTypeSelectionMove,
    AXTextStateChangeTypeSelectionExtend,
    AXTextStateChangeTypeSelectionBoundary
};

enum AXTextEditType {
    AXTextEditTypeUnknown,
    AXTextEditTypeDelete, // Generic text delete
    AXTextEditTypeInsert, // Generic text insert
    AXTextEditTypeTyping, // Insert via typing
    AXTextEditTypeDictation, // Insert via dictation
    AXTextEditTypeCut, // Delete via Cut
    AXTextEditTypePaste, // Insert via Paste
    AXTextEditTypeAttributesChange // Change font, style, alignment, color, etc.
};

enum AXTextSelectionDirection {
    AXTextSelectionDirectionUnknown,
    AXTextSelectionDirectionBeginning,
    AXTextSelectionDirectionEnd,
    AXTextSelectionDirectionPrevious,
    AXTextSelectionDirectionNext,
    AXTextSelectionDirectionDiscontiguous
};

enum AXTextSelectionGranularity {
    AXTextSelectionGranularityUnknown,
    AXTextSelectionGranularityCharacter,
    AXTextSelectionGranularityWord,
    AXTextSelectionGranularityLine,
    AXTextSelectionGranularitySentence,
    AXTextSelectionGranularityParagraph,
    AXTextSelectionGranularityPage,
    AXTextSelectionGranularityDocument,
    AXTextSelectionGranularityAll // All granularity represents the action of selecting the whole document as a single action. Extending selection by some other granularity until it encompasses the whole document will not result in a all granularity notification.
};

struct AXTextSelection {
    AXTextSelectionDirection direction;
    AXTextSelectionGranularity granularity;
    bool focusChange;
};

struct AXTextStateChangeIntent {
    AXTextStateChangeType type;
    union {
        AXTextSelection selection;
        AXTextEditType change;
    };

    AXTextStateChangeIntent(AXTextStateChangeType type = AXTextStateChangeTypeUnknown, AXTextSelection selection = AXTextSelection())
        : type(type)
        , selection(selection)
    { }

    AXTextStateChangeIntent(AXTextEditType change)
        : type(AXTextStateChangeTypeEdit)
        , change(change)
    { }
};

} // namespace WebCore
