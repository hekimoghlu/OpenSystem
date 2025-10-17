/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 21, 2022.
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

#include "FloatRect.h"
#include <wtf/RefCounted.h>
#include <wtf/RefPtr.h>

namespace WebCore {

class CanvasRenderingContext2DBase;
class Filter;
class GraphicsContext;
class GraphicsContextSwitcher;

class CanvasLayerContextSwitcher : public RefCounted<CanvasLayerContextSwitcher> {
public:
    static RefPtr<CanvasLayerContextSwitcher> create(CanvasRenderingContext2DBase&, const FloatRect& bounds, RefPtr<Filter>&&);

    ~CanvasLayerContextSwitcher();

    GraphicsContext* drawingContext() const;
    FloatRect expandedBounds() const { return m_bounds + outsets(); }

private:
    CanvasLayerContextSwitcher(CanvasRenderingContext2DBase&, const FloatRect& bounds, std::unique_ptr<GraphicsContextSwitcher>&&);

    FloatBoxExtent outsets() const;
    Ref<CanvasRenderingContext2DBase> protectedContext() const { return m_context.get(); }

    WeakRef<CanvasRenderingContext2DBase> m_context;
    GraphicsContext* m_effectiveDrawingContext;
    FloatRect m_bounds;
    std::unique_ptr<GraphicsContextSwitcher> m_targetSwitcher;
};

} // namespace WebCore
