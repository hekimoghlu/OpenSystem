/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 14, 2023.
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
#include "ClipStack.h"

#include <array>
#include <wtf/StdLibExtras.h>
#include "TextureMapperGLHeaders.h"

namespace WebCore {

void ClipStack::push()
{
    clipStack.append(clipState);
    clipStateDirty = true;
}

void ClipStack::pop()
{
    if (clipStack.isEmpty())
        return;
    clipState = clipStack.last();
    clipStack.removeLast();
    clipStateDirty = true;
}

void ClipStack::reset(const IntRect& rect, ClipStack::YAxisMode mode)
{
    clipStack.clear();
    size = rect.size();
    yAxisMode = mode;
    clipState = State(rect);
    clipStateDirty = true;
}

void ClipStack::intersect(const IntRect& rect)
{
    clipState.scissorBox.intersect(rect);
    clipStateDirty = true;
}

void ClipStack::setStencilIndex(int stencilIndex)
{
    clipState.stencilIndex = stencilIndex;
    clipStateDirty = true;
}

void ClipStack::addRoundedRect(const FloatRoundedRect& roundedRect, const TransformationMatrix& matrix)
{
    if (clipState.roundedRectCount >= s_roundedRectMaxClips)
        return;

    // Ensure that the vectors holding the components have the required size.
    m_roundedRectComponents.grow(s_roundedRectComponentsArraySize);
    m_roundedRectInverseTransformComponents.grow(s_roundedRectInverseTransformComponentsArraySize);

    // Copy the RoundedRect components to the appropriate position in the array.
    int basePosition = clipState.roundedRectCount * s_roundedRectComponentsPerRect;
    m_roundedRectComponents[basePosition] = roundedRect.rect().x();
    m_roundedRectComponents[basePosition + 1] = roundedRect.rect().y();
    m_roundedRectComponents[basePosition + 2] = roundedRect.rect().width();
    m_roundedRectComponents[basePosition + 3] = roundedRect.rect().height();
    m_roundedRectComponents[basePosition + 4] = roundedRect.radii().topLeft().width();
    m_roundedRectComponents[basePosition + 5] = roundedRect.radii().topLeft().height();
    m_roundedRectComponents[basePosition + 6] = roundedRect.radii().topRight().width();
    m_roundedRectComponents[basePosition + 7] = roundedRect.radii().topRight().height();
    m_roundedRectComponents[basePosition + 8] = roundedRect.radii().bottomLeft().width();
    m_roundedRectComponents[basePosition + 9] = roundedRect.radii().bottomLeft().height();
    m_roundedRectComponents[basePosition + 10] = roundedRect.radii().bottomRight().width();
    m_roundedRectComponents[basePosition + 11] = roundedRect.radii().bottomRight().height();

    // Copy the TransformationMatrix components to the appropriate position in the array.
    basePosition = clipState.roundedRectCount * s_roundedRectInverseTransformComponentsPerRect;
    memcpySpan(m_roundedRectInverseTransformComponents.mutableSpan().subspan(basePosition), std::span<const float, 16> { matrix.toColumnMajorFloatArray() }.first(s_roundedRectInverseTransformComponentsPerRect));

    clipState.roundedRectCount++;
}

void ClipStack::apply()
{
    if (clipState.scissorBox.isEmpty())
        return;

    glScissor(clipState.scissorBox.x(),
        (yAxisMode == YAxisMode::Inverted) ? size.height() - clipState.scissorBox.maxY() : clipState.scissorBox.y(),
        clipState.scissorBox.width(), clipState.scissorBox.height());
    glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP);
    glStencilFunc(GL_EQUAL, clipState.stencilIndex - 1, clipState.stencilIndex - 1);
    if (clipState.stencilIndex == 1)
        glDisable(GL_STENCIL_TEST);
    else
        glEnable(GL_STENCIL_TEST);
}

void ClipStack::applyIfNeeded()
{
    if (!clipStateDirty)
        return;

    clipStateDirty = false;
    apply();
}

} // namespace WebCore
