/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 5, 2025.
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

#include "DictationAlternative.h"
#include "TextInsertionBaseCommand.h"
#include <wtf/CheckedRef.h>

namespace WebCore {

class DictationCommand : public TextInsertionBaseCommand {
    friend class DictationCommandLineOperation;
public:
    static void insertText(Ref<Document>&&, const String&, const Vector<DictationAlternative>& alternatives, const VisibleSelection&);
    bool isDictationCommand() const override { return true; }
private:
    static Ref<DictationCommand> create(Ref<Document>&& document, const String& text, const Vector<DictationAlternative>& alternatives)
    {
        return adoptRef(*new DictationCommand(WTFMove(document), text, alternatives));
    }

    DictationCommand(Ref<Document>&&, const String& text, const Vector<DictationAlternative>& alternatives);
    
    void doApply() override;

    void insertTextRunWithoutNewlines(size_t lineStart, size_t lineLength);
    void insertParagraphSeparator();
    void collectDictationAlternativesInRange(size_t rangeStart, size_t rangeLength, Vector<DictationAlternative>&);

    String m_textToInsert;
    Vector<DictationAlternative> m_alternatives;
};

} // namespace WebCore
