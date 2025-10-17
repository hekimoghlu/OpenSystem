/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 8, 2023.
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
#include "EditAction.h"

namespace WebCore {

class InsertNestedListCommand final : public CompositeEditCommand {
public:
    static void insertUnorderedList(Document&);
    static void insertOrderedList(Document&);

private:
    enum class Type : uint8_t { OrderedList, UnorderedList };

    static Ref<InsertNestedListCommand> create(Ref<Document>&& document, Type type)
    {
        return adoptRef(*new InsertNestedListCommand(WTFMove(document), type));
    }

    InsertNestedListCommand(Ref<Document>&& document, Type type)
        : CompositeEditCommand(WTFMove(document))
        , m_type(type)
    {
    }

    EditAction editingAction() const final
    {
        if (m_type == Type::OrderedList)
            return EditAction::InsertOrderedList;

        return EditAction::InsertUnorderedList;
    }

    bool preservesTypingStyle() const final { return true; }
    void doApply() final;

    Type m_type;
};

} // namespace WebCore
