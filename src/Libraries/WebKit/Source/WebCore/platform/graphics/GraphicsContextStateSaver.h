/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 9, 2021.
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

#include "GraphicsContext.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

class GraphicsContextStateSaver {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(GraphicsContextStateSaver);
public:
    GraphicsContextStateSaver(GraphicsContext& context, bool saveAndRestore = true)
        : m_context(context)
        , m_saveAndRestore(saveAndRestore)
    {
        if (m_saveAndRestore)
            m_context.save();
    }
    
    ~GraphicsContextStateSaver()
    {
        if (m_saveAndRestore)
            m_context.restore();
    }
    
    void save()
    {
        ASSERT(!m_saveAndRestore);
        m_context.save();
        m_saveAndRestore = true;
    }

    void restore()
    {
        ASSERT(m_saveAndRestore);
        m_context.restore();
        m_saveAndRestore = false;
    }
    
    GraphicsContext* context() const { return &m_context; }

private:
    GraphicsContext& m_context;
    bool m_saveAndRestore;
};

class TransparencyLayerScope {
public:
    TransparencyLayerScope(GraphicsContext& context, float alpha, bool beginLayer = true)
        : m_context(context)
        , m_alpha(alpha)
        , m_beganLayer(beginLayer)
    {
        if (beginLayer)
            m_context.beginTransparencyLayer(m_alpha);
    }
    
    void beginLayer(float alpha)
    {
        m_alpha = alpha;
        m_context.beginTransparencyLayer(m_alpha);
        m_beganLayer = true;
    }
    
    ~TransparencyLayerScope()
    {
        if (m_beganLayer)
            m_context.endTransparencyLayer();
    }

private:
    GraphicsContext& m_context;
    float m_alpha;
    bool m_beganLayer;
};

class GraphicsContextStateStackChecker {
public:
    GraphicsContextStateStackChecker(GraphicsContext& context)
        : m_context(context)
        , m_stackSize(context.stackSize())
    { }
    
    ~GraphicsContextStateStackChecker()
    {
        if (m_context.stackSize() != m_stackSize)
            WTFLogAlways("GraphicsContext %p stack changed by %d", &m_context, (int)m_context.stackSize() - (int)m_stackSize);
    }

private:
    GraphicsContext& m_context;
    unsigned m_stackSize;
};

class InterpolationQualityMaintainer {
public:
    explicit InterpolationQualityMaintainer(GraphicsContext& graphicsContext, InterpolationQuality interpolationQualityToUse)
        : m_graphicsContext(graphicsContext)
        , m_currentInterpolationQuality(graphicsContext.imageInterpolationQuality())
        , m_interpolationQualityChanged(interpolationQualityToUse != InterpolationQuality::Default && m_currentInterpolationQuality != interpolationQualityToUse)
    {
        if (m_interpolationQualityChanged)
            m_graphicsContext.setImageInterpolationQuality(interpolationQualityToUse);
    }

    explicit InterpolationQualityMaintainer(GraphicsContext& graphicsContext, std::optional<InterpolationQuality> interpolationQuality)
        : InterpolationQualityMaintainer(graphicsContext, interpolationQuality ? interpolationQuality.value() : graphicsContext.imageInterpolationQuality())
    {
    }

    ~InterpolationQualityMaintainer()
    {
        if (m_interpolationQualityChanged)
            m_graphicsContext.setImageInterpolationQuality(m_currentInterpolationQuality);
    }

private:
    GraphicsContext& m_graphicsContext;
    InterpolationQuality m_currentInterpolationQuality;
    bool m_interpolationQualityChanged;
};

} // namespace WebCore
