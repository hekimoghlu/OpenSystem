/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 23, 2023.
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

#include "PseudoElementIdentifier.h"
#include "RenderStyleConstants.h"
#include "StyleScrollbarState.h"
#include <wtf/text/AtomString.h>

namespace WebCore::Style {

class PseudoElementRequest {
public:
    PseudoElementRequest(PseudoId pseudoId, std::optional<StyleScrollbarState> scrollbarState = std::nullopt)
        : m_identifier({ pseudoId })
        , m_scrollbarState(scrollbarState)
    {
        ASSERT(pseudoId != PseudoId::None);
    }

    PseudoElementRequest(PseudoId pseudoId, const AtomString& nameArgument)
        : m_identifier({ pseudoId, nameArgument })
    {
        ASSERT(pseudoId == PseudoId::Highlight || pseudoId == PseudoId::ViewTransitionGroup || pseudoId == PseudoId::ViewTransitionImagePair || pseudoId == PseudoId::ViewTransitionOld || pseudoId == PseudoId::ViewTransitionNew);
    }

    PseudoElementRequest(const PseudoElementIdentifier& pseudoElementIdentifier)
        : m_identifier(pseudoElementIdentifier)
    {
        ASSERT(pseudoElementIdentifier.pseudoId != PseudoId::None);
    }

    const PseudoElementIdentifier& identifier() const { return m_identifier; }
    PseudoId pseudoId() const { return m_identifier.pseudoId; }
    const AtomString& nameArgument() const { return m_identifier.nameArgument; }
    const std::optional<StyleScrollbarState>& scrollbarState() const { return m_scrollbarState; }

private:
    PseudoElementIdentifier m_identifier;
    std::optional<StyleScrollbarState> m_scrollbarState;
};

} // namespace WebCore
