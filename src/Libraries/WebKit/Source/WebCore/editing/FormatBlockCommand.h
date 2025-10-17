/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 19, 2025.
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
#include "QualifiedName.h"

namespace WebCore {

class Document;
class Element;
class Position;
class Range;
class VisiblePosition;

class FormatBlockCommand : public ApplyBlockElementCommand {
public:
    static Ref<FormatBlockCommand> create(Ref<Document>&& document, const QualifiedName& tagName)
    {
        return adoptRef(*new FormatBlockCommand(WTFMove(document), tagName));
    }
    
    bool preservesTypingStyle() const override { return true; }

    static RefPtr<Element> elementForFormatBlockCommand(const std::optional<SimpleRange>&);
    bool didApply() const { return m_didApply; }

private:
    FormatBlockCommand(Ref<Document>&&, const QualifiedName& tagName);

    void formatSelection(const VisiblePosition& startOfSelection, const VisiblePosition& endOfSelection) override;
    void formatRange(const Position& start, const Position& end, const Position& endOfSelection, RefPtr<Element>&) override;
    EditAction editingAction() const override { return EditAction::FormatBlock; }

    bool m_didApply;
};

} // namespace WebCore
