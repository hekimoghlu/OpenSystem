/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 18, 2025.
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
#include "SplitTextNodeCommand.h"

#include "CompositeEditCommand.h"
#include "Document.h"
#include "DocumentInlines.h"
#include "DocumentMarkerController.h"
#include "Text.h"
#include <wtf/Assertions.h>

namespace WebCore {

SplitTextNodeCommand::SplitTextNodeCommand(Ref<Text>&& text, int offset)
    : SimpleEditCommand(text->document())
    , m_text2(WTFMove(text))
    , m_offset(offset)
{
    // NOTE: Various callers rely on the fact that the original node becomes
    // the second node (i.e. the new node is inserted before the existing one).
    // That is not a fundamental dependency (i.e. it could be re-coded), but
    // rather is based on how this code happens to work.
    ASSERT(m_text2->length() > 0);
    ASSERT(m_offset > 0);
    ASSERT(m_offset < m_text2->length());
}

void SplitTextNodeCommand::doApply()
{
    RefPtr parent = m_text2->parentNode();
    if (!parent || !parent->hasEditableStyle())
        return;

    auto result = protectedText2()->substringData(0, m_offset);
    if (result.hasException())
        return;
    auto prefixText = result.releaseReturnValue();
    if (prefixText.isEmpty())
        return;

    m_text1 = Text::create(document(), WTFMove(prefixText));
    ASSERT(m_text1);
    if (CheckedPtr markers = document().markersIfExists())
        markers->copyMarkers(protectedText2(), { 0, m_offset }, *protectedText1());

    insertText1AndTrimText2();
}

void SplitTextNodeCommand::doUnapply()
{
    RefPtr text1 = m_text1;
    if (!text1 || !text1->hasEditableStyle())
        return;

    ASSERT(&text1->document() == &document());

    String prefixText = text1->data();

    Ref text2 = m_text2;
    text2->insertData(0, prefixText);

    if (CheckedPtr markers = document().markersIfExists())
        markers->copyMarkers(*text1, { 0, prefixText.length() }, text2);
    text1->remove();
}

void SplitTextNodeCommand::doReapply()
{
    if (!m_text1)
        return;

    RefPtr parent = m_text2->parentNode();
    if (!parent || !parent->hasEditableStyle())
        return;

    insertText1AndTrimText2();
}

void SplitTextNodeCommand::insertText1AndTrimText2()
{
    Ref text2 = m_text2;
    if (text2->parentNode()->insertBefore(*m_text1, text2.copyRef()).hasException())
        return;
    text2->deleteData(0, m_offset);
}

#ifndef NDEBUG

void SplitTextNodeCommand::getNodesInCommand(NodeSet& nodes)
{
    addNodeAndDescendants(protectedText1().get(), nodes);
    addNodeAndDescendants(protectedText2().ptr(), nodes);
}

#endif
    
} // namespace WebCore
