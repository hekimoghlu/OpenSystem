/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 13, 2022.
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

#include "EditCommand.h"

namespace WebCore {

class Text;

class InsertIntoTextNodeCommand : public SimpleEditCommand {
public:
    static Ref<InsertIntoTextNodeCommand> create(Ref<Text>&& node, unsigned offset, const String& text, EditAction editingAction = EditAction::Insert)
    {
        return adoptRef(*new InsertIntoTextNodeCommand(WTFMove(node), offset, text, editingAction));
    }

    const String& insertedText();

protected:
    InsertIntoTextNodeCommand(Ref<Text>&& node, unsigned offset, const String& text, EditAction editingAction);

private:
    void doApply() override;
    void doUnapply() override;
    void doReapply() override;

    Ref<Text> protectedNode() const { return m_node.get(); }
    
#ifndef NDEBUG
    void getNodesInCommand(NodeSet&) override;
#endif
    
    Ref<Text> m_node;
    unsigned m_offset;
    String m_text;
};

inline const String& InsertIntoTextNodeCommand::insertedText()
{
    return m_text;
}

} // namespace WebCore
