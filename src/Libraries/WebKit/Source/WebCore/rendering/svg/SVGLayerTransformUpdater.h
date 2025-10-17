/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 3, 2023.
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

#include "RenderElementInlines.h"
#include "RenderLayerModelObject.h"
#include <wtf/WeakPtr.h>

namespace WebCore {

class SVGLayerTransformUpdater {
    WTF_MAKE_NONCOPYABLE(SVGLayerTransformUpdater);
public:
    SVGLayerTransformUpdater(RenderLayerModelObject& renderer)
        : m_renderer(renderer)
    {
        if (!m_renderer->hasLayer())
            return;

        m_transformReferenceBox = m_renderer->transformReferenceBoxRect();
        m_layerTransform = m_renderer->layerTransform();

        m_renderer->updateLayerTransform();
    }

    ~SVGLayerTransformUpdater()
    {
        if (!m_renderer->hasLayer())
            return;
        if (m_renderer->transformReferenceBoxRect() == m_transformReferenceBox)
            return;

        m_renderer->updateLayerTransform();
    }

    bool layerTransformChanged() const
    {
        auto* layerTransform = m_renderer->layerTransform();

        bool hasTransform = !!layerTransform;
        bool hadTransform = !!m_layerTransform;
        if (hasTransform != hadTransform)
            return true;

        return hasTransform && (*layerTransform != *m_layerTransform);
    }

private:
    SingleThreadWeakRef<RenderLayerModelObject> m_renderer;
    FloatRect m_transformReferenceBox;
    TransformationMatrix* m_layerTransform { nullptr };
};

} // namespace WebCore

