/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 11, 2023.
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

#include "CompositeEditCommand.h"
#include "QualifiedName.h"

namespace WebCore {

class ApplyBlockElementCommand : public CompositeEditCommand {
protected:
    ApplyBlockElementCommand(Ref<Document>&&, const QualifiedName& tagName, const AtomString& inlineStyle);
    ApplyBlockElementCommand(Ref<Document>&&, const QualifiedName& tagName);

    virtual void formatSelection(const VisiblePosition& startOfSelection, const VisiblePosition& endOfSelection);
    Ref<HTMLElement> createBlockElement();
    const QualifiedName tagName() const { return m_tagName; }

private:
    void doApply() override;
    virtual void formatRange(const Position& start, const Position& end, const Position& endOfSelection, RefPtr<Element>&) = 0;
    const RenderStyle* renderStyleOfEnclosingTextNode(const Position&);
    void rangeForParagraphSplittingTextNodesIfNeeded(const VisiblePosition&, Position&, Position&);
    VisiblePosition endOfNextParagraphSplittingTextNodesIfNeeded(VisiblePosition&, Position&, Position&);

    QualifiedName m_tagName;
    AtomString m_inlineStyle;
    Position m_endOfLastParagraph;
};

} // namespace WebCore
