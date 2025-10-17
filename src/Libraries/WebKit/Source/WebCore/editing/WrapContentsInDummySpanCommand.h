/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 9, 2024.
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

class HTMLElement;

class WrapContentsInDummySpanCommand : public SimpleEditCommand {
public:
    static Ref<WrapContentsInDummySpanCommand> create(Element& element)
    {
        return adoptRef(*new WrapContentsInDummySpanCommand(element));
    }

private:
    explicit WrapContentsInDummySpanCommand(Element&);

    void doApply() override;
    void doUnapply() override;
    void doReapply() override;
    void executeApply();

    RefPtr<HTMLElement> protectedDummySpan() const { return m_dummySpan; }
    Ref<Element> protectedElement() const { return m_element; }

#ifndef NDEBUG
    void getNodesInCommand(NodeSet&) override;
#endif

    Ref<Element> m_element;
    RefPtr<HTMLElement> m_dummySpan;
};

} // namespace WebCore
