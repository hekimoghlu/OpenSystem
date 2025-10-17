/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 30, 2023.
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

struct AttributedString;

// A Writing Tools Composition command is essentially a wrapper around a group of Replace Selection commands,
// which can be undone/redone as a whole.
//
// It also maintains an associated context range that persists after each constituent command.
class WritingToolsCompositionCommand : public CompositeEditCommand {
public:
    enum class MatchStyle: bool {
        No, Yes
    };

    enum class State: uint8_t {
        InProgress,
        Complete
    };

    static Ref<WritingToolsCompositionCommand> create(Ref<Document>&& document, const SimpleRange& endingContextRange)
    {
        return adoptRef(*new WritingToolsCompositionCommand(WTFMove(document), endingContextRange));
    }

    // This method is used to add each "piece" (aka Replace selection command) of the Writing Tools composition command.
    void replaceContentsOfRangeWithFragment(RefPtr<DocumentFragment>&&, const SimpleRange&, MatchStyle, State);

    SimpleRange endingContextRange() const { return m_endingContextRange; }

    // FIXME: Remove this when WritingToolsController no longer needs to support `contextRangeForSessionWithID`.
    SimpleRange currentContextRange() const { return m_currentContextRange; }

    void commit();

private:
    WritingToolsCompositionCommand(Ref<Document>&&, const SimpleRange&);

    void doApply() override { }

    SimpleRange m_endingContextRange;
    SimpleRange m_currentContextRange;
};

} // namespace WebCore
