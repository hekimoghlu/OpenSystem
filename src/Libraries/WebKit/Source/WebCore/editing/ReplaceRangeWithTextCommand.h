/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 7, 2022.
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

class DocumentFragment;

class ReplaceRangeWithTextCommand : public CompositeEditCommand {
public:
    static Ref<ReplaceRangeWithTextCommand> create(const SimpleRange& rangeToBeReplaced, const String& text)
    {
        return adoptRef(*new ReplaceRangeWithTextCommand(rangeToBeReplaced, text));
    }

private:
    ReplaceRangeWithTextCommand(const SimpleRange& rangeToBeReplaced, const String& text);
    bool willApplyCommand() final;
    void doApply() override;
    String inputEventData() const final;
    RefPtr<DataTransfer> inputEventDataTransfer() const final;
    Vector<RefPtr<StaticRange>> targetRanges() const final;

    RefPtr<DocumentFragment> protectedTextFragment() const { return m_textFragment; }

    SimpleRange m_rangeToBeReplaced;
    RefPtr<DocumentFragment> m_textFragment;
    String m_text;
};

} // namespace WebCore
