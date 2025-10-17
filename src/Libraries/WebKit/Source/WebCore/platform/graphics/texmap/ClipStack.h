/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 14, 2025.
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
#ifndef ClipStack_h
#define ClipStack_h

#include "FloatRoundedRect.h"
#include "IntRect.h"
#include "IntSize.h"
#include "TransformationMatrix.h"
#include <wtf/Vector.h>

namespace WebCore {

// Because GLSL uniform arrays need to have a defined size, we need to put a limit to the number of simultaneous
// rounded rectangle clips that we're going to allow. Currently this is defined to 10.
// This value must be kept in sync with the sizes defined in TextureMapperShaderProgram.cpp.
static const unsigned s_roundedRectMaxClips = 10;

// When converting a rounded rectangle to an array of floats, we need 12 elements. So the size of the array
// required to contain the 10 rectangles is 12 * 10 = 120.
// This value must be kept in sync with the sizes defined in TextureMapperShaderProgram.cpp.
static const unsigned s_roundedRectComponentsPerRect = 12;
static const unsigned s_roundedRectComponentsArraySize = s_roundedRectMaxClips * s_roundedRectComponentsPerRect;

// When converting a transformation matrix to an array of floats, we need 16 elements. So the size of the array
// required to contain the 10 matrices is 16 * 10 = 160.
// This value must be kept in sync with the sizes defined in TextureMapperShaderProgram.cpp.
static const unsigned s_roundedRectInverseTransformComponentsPerRect = 16;
static const unsigned s_roundedRectInverseTransformComponentsArraySize = s_roundedRectMaxClips * s_roundedRectInverseTransformComponentsPerRect;

class ClipStack {
public:
    struct State {
        explicit State(const IntRect& scissors = IntRect())
            : scissorBox(scissors)
        { }

        IntRect scissorBox;
        int stencilIndex { 1 };
        unsigned roundedRectCount { 0 };
    };

    // Y-axis should be inverted only when painting into the window.
    enum class YAxisMode {
        Default,
        Inverted,
    };

    void push();
    void pop();
    State& current() { return clipState; }

    void reset(const IntRect&, YAxisMode);
    void intersect(const IntRect&);
    void setStencilIndex(int);
    int getStencilIndex() const { return clipState.stencilIndex; }

    void addRoundedRect(const FloatRoundedRect&, const TransformationMatrix&);
    const float* roundedRectComponents() const { return m_roundedRectComponents.data(); }
    const float* roundedRectInverseTransformComponents() const { return m_roundedRectInverseTransformComponents.data(); }
    unsigned roundedRectCount() const { return clipState.roundedRectCount; }
    bool isRoundedRectClipEnabled() const { return !!clipState.roundedRectCount; }
    bool isRoundedRectClipAllowed() const { return clipState.roundedRectCount < s_roundedRectMaxClips; }

    void apply();
    void applyIfNeeded();

    bool isCurrentScissorBoxEmpty() const { return clipState.scissorBox.isEmpty(); }

private:
    Vector<State> clipStack;
    State clipState;
    IntSize size;
    bool clipStateDirty { false };
    YAxisMode yAxisMode { YAxisMode::Default };
    Vector<float, s_roundedRectComponentsArraySize> m_roundedRectComponents;
    Vector<float, s_roundedRectInverseTransformComponentsArraySize> m_roundedRectInverseTransformComponents;
};

} // namespace WebCore

#endif // ClipStack_h
