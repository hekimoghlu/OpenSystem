/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 3, 2024.
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
#include "SplitTextNodeContainingElementCommand.h"

#include "Element.h"
#include "ElementInlines.h"
#include "RenderElement.h"
#include "Text.h"
#include <wtf/Assertions.h>

namespace WebCore {

SplitTextNodeContainingElementCommand::SplitTextNodeContainingElementCommand(Ref<Text>&& text, int offset)
    : CompositeEditCommand(text->document())
    , m_text(WTFMove(text))
    , m_offset(offset)
{
    ASSERT(m_text->length() > 0);
}

void SplitTextNodeContainingElementCommand::doApply()
{
    ASSERT(m_offset > 0);

    splitTextNode(m_text, m_offset);

    RefPtr parent = m_text->parentElement();
    if (!parent || !parent->parentElement() || !parent->parentElement()->hasEditableStyle())
        return;

    bool parentRendererIsNoneOrNotInline = false;
    {
        CheckedPtr parentRenderer = parent->renderer();
        parentRendererIsNoneOrNotInline = !parentRenderer || !parentRenderer->isInline();
    }
    if (parentRendererIsNoneOrNotInline) {
        wrapContentsInDummySpan(*parent);
        RefPtr firstChild = parent->firstChild();
        if (!is<Element>(firstChild))
            return;
        parent = downcast<Element>(WTFMove(firstChild));
    }

    splitElement(*parent, m_text);
}

}
