/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 6, 2022.
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
#include "ElementAnimationRareData.h"

#include "CSSAnimation.h"
#include "CSSTransition.h"
#include "KeyframeEffectStack.h"
#include "RenderStyle.h"
#include "ScriptExecutionContext.h"

namespace WebCore {
DEFINE_ALLOCATOR_WITH_HEAP_IDENTIFIER(ElementAnimationRareData);

ElementAnimationRareData::ElementAnimationRareData()
{
}

ElementAnimationRareData::~ElementAnimationRareData()
{
}

KeyframeEffectStack& ElementAnimationRareData::ensureKeyframeEffectStack()
{
    if (!m_keyframeEffectStack)
        m_keyframeEffectStack = makeUnique<KeyframeEffectStack>();
    return *m_keyframeEffectStack.get();
}

void ElementAnimationRareData::setAnimationsCreatedByMarkup(CSSAnimationCollection&& animations)
{
    m_animationsCreatedByMarkup = WTFMove(animations);
}

void ElementAnimationRareData::setLastStyleChangeEventStyle(std::unique_ptr<const RenderStyle>&& style)
{
    if (m_keyframeEffectStack && m_lastStyleChangeEventStyle != style) {
        auto previousStyleChangeEventStyle = std::exchange(m_lastStyleChangeEventStyle, WTFMove(style));
        m_keyframeEffectStack->lastStyleChangeEventStyleDidChange(previousStyleChangeEventStyle.get(), m_lastStyleChangeEventStyle.get());
        return;
    }

    m_lastStyleChangeEventStyle = WTFMove(style);
}

} // namespace WebCore
