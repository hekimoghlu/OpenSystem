/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 8, 2025.
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

class MergeIdenticalElementsCommand : public SimpleEditCommand {
public:
    static Ref<MergeIdenticalElementsCommand> create(Ref<Element>&& element1, Ref<Element>&& element2)
    {
        return adoptRef(*new MergeIdenticalElementsCommand(WTFMove(element1), WTFMove(element2)));
    }

private:
    MergeIdenticalElementsCommand(Ref<Element>&&, Ref<Element>&&);

    void doApply() override;
    void doUnapply() override;
    
#ifndef NDEBUG
    void getNodesInCommand(NodeSet&) override;
#endif

    Ref<Element> protectedElement1() const { return m_element1; }
    Ref<Element> protectedElement2() const { return m_element2; }
    
    Ref<Element> m_element1;
    Ref<Element> m_element2;
    RefPtr<Node> m_atChild;
};

} // namespace WebCore
