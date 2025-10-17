/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 1, 2024.
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

class SplitTextNodeCommand : public SimpleEditCommand {
public:
    static Ref<SplitTextNodeCommand> create(Ref<Text>&& node, int offset)
    {
        return adoptRef(*new SplitTextNodeCommand(WTFMove(node), offset));
    }

private:
    SplitTextNodeCommand(Ref<Text>&&, int offset);

    void doApply() override;
    void doUnapply() override;
    void doReapply() override;
    void insertText1AndTrimText2();
    
#ifndef NDEBUG
    void getNodesInCommand(NodeSet&) override;
#endif

    RefPtr<Text> protectedText1() const { return m_text1; }
    Ref<Text> protectedText2() const { return m_text2; }

    RefPtr<Text> m_text1;
    Ref<Text> m_text2;
    unsigned m_offset;
};

} // namespace WebCore
