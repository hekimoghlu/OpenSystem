/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 31, 2024.
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

#include "ApplyBlockElementCommand.h"
#include "EditAction.h"

namespace WebCore {

class IndentOutdentCommand : public ApplyBlockElementCommand {
public:
    enum EIndentType { Indent, Outdent };
    static Ref<IndentOutdentCommand> create(Ref<Document>&& document, EIndentType type)
    {
        return adoptRef(*new IndentOutdentCommand(WTFMove(document), type));
    }

    bool preservesTypingStyle() const override { return true; }

private:
    IndentOutdentCommand(Ref<Document>&&, EIndentType);

    EditAction editingAction() const override { return m_typeOfAction == Indent ? EditAction::Indent : EditAction::Outdent; }

    void outdentRegion(const VisiblePosition&, const VisiblePosition&);
    void outdentParagraph();
    bool tryIndentingAsListItem(const Position&, const Position&);
    void indentIntoBlockquote(const Position&, const Position&, RefPtr<Element>&);

    void formatSelection(const VisiblePosition& startOfSelection, const VisiblePosition& endOfSelection) override;
    void formatRange(const Position& start, const Position& end, const Position& endOfSelection, RefPtr<Element>& blockquoteForNextIndent) override;

    EIndentType m_typeOfAction;
};

} // namespace WebCore
