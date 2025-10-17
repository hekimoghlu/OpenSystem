/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 15, 2022.
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
#include <wtf/text/WTFString.h>

namespace WebCore {

class Document;
class VisibleSelection;

class TextInsertionBaseCommand : public CompositeEditCommand {
public:
    virtual ~TextInsertionBaseCommand() { }

protected:
    explicit TextInsertionBaseCommand(Ref<Document>&&, EditAction = EditAction::Unspecified);
    static void applyTextInsertionCommand(LocalFrame*, TextInsertionBaseCommand&, const VisibleSelection& selectionForInsertion, const VisibleSelection& endingSelection);
};

String dispatchBeforeTextInsertedEvent(const String& text, const VisibleSelection& selectionForInsertion, bool insertionIsForUpdatingComposition);
bool canAppendNewLineFeedToSelection(const VisibleSelection&);

// LineOperation should define member function "opeartor (size_t lineOffset, size_t lineLength, bool isLastLine)".
// lienLength doesn't include the newline character. So the value of lineLength could be 0.
template <class LineOperation>
void forEachLineInString(const String& string, const LineOperation& operation)
{
    unsigned offset = 0;
    size_t newline;
    while ((newline = string.find('\n', offset)) != notFound) {
        operation(offset, newline - offset, false);
        offset = newline + 1;
    }
    if (!offset)
        operation(0, string.length(), true);
    else {
        unsigned length = string.length();
        if (length != offset)
            operation(offset, length - offset, true);
    }
}

} // namespace WebCore
