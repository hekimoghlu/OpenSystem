/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 31, 2025.
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

class HTMLElement;

// More accurately, this is ReplaceElementWithSpanPreservingChildrenAndAttributesCommand
class ReplaceNodeWithSpanCommand : public SimpleEditCommand {
public:
    static Ref<ReplaceNodeWithSpanCommand> create(Ref<HTMLElement>&& element)
    {
        return adoptRef(*new ReplaceNodeWithSpanCommand(WTFMove(element)));
    }

    HTMLElement* spanElement() { return m_spanElement.get(); }

private:
    explicit ReplaceNodeWithSpanCommand(Ref<HTMLElement>&&);

    void doApply() override;
    void doUnapply() override;

    RefPtr<HTMLElement> protectedSpanElement() const { return m_spanElement; }
    Ref<HTMLElement> protectedElementToReplace() const { return m_elementToReplace; }
    
#ifndef NDEBUG
    void getNodesInCommand(NodeSet&) override;
#endif

    Ref<HTMLElement> m_elementToReplace;
    RefPtr<HTMLElement> m_spanElement;
};

} // namespace WebCore
