/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 25, 2023.
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

namespace WebCore {

class DocumentMarkerController;
class Text;

class TextInsertionMarkerSupplier : public RefCounted<TextInsertionMarkerSupplier> {
public:
    virtual ~TextInsertionMarkerSupplier() = default;
    virtual void addMarkersToTextNode(Text&, unsigned offsetOfInsertion, const String& textInserted) = 0;
protected:
    TextInsertionMarkerSupplier() = default;
};

class InsertTextCommand : public CompositeEditCommand {
public:
    enum RebalanceType {
        RebalanceLeadingAndTrailingWhitespaces,
        RebalanceAllWhitespaces
    };

    static Ref<InsertTextCommand> create(Ref<Document>&& document, const String& text, bool selectInsertedText = false,
        RebalanceType rebalanceType = RebalanceLeadingAndTrailingWhitespaces, EditAction editingAction = EditAction::Insert)
    {
        return adoptRef(*new InsertTextCommand(WTFMove(document), text, selectInsertedText, rebalanceType, editingAction));
    }

    static Ref<InsertTextCommand> createWithMarkerSupplier(Ref<Document>&& document, const String& text, Ref<TextInsertionMarkerSupplier>&& markerSupplier, EditAction editingAction = EditAction::Insert)
    {
        return adoptRef(*new InsertTextCommand(WTFMove(document), text, WTFMove(markerSupplier), editingAction));
    }

protected:
    InsertTextCommand(Ref<Document>&&, const String& text, Ref<TextInsertionMarkerSupplier>&&, EditAction);
    InsertTextCommand(Ref<Document>&&, const String& text, bool selectInsertedText, RebalanceType, EditAction);

private:

    void doApply() override;

    bool isInsertTextCommand() const override { return true; }

    Position positionInsideTextNode(const Position&);
    Position insertTab(const Position&);
    
    bool performTrivialReplace(const String&, bool selectInsertedText);
    bool performOverwrite(const String&, bool selectInsertedText);
    void setEndingSelectionWithoutValidation(const Position& startPosition, const Position& endPosition);

    friend class TypingCommand;

    String m_text;
    bool m_selectInsertedText;
    RebalanceType m_rebalanceType;
    RefPtr<TextInsertionMarkerSupplier> m_markerSupplier;
};

} // namespace WebCore
