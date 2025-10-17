/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 17, 2021.
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

#if ENABLE(SERVICE_CONTROLS)

#include <wtf/CheckedRef.h>
#include <wtf/Noncopyable.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>
#include <wtf/WeakRef.h>

namespace WebCore {

class FloatQuad;
class LayoutRect;
class RenderLayerModelObject;
class RenderView;

struct GapRects;

class SelectionGeometryGatherer {
    WTF_MAKE_TZONE_ALLOCATED(SelectionGeometryGatherer);
    WTF_MAKE_NONCOPYABLE(SelectionGeometryGatherer);

public:
    SelectionGeometryGatherer(RenderView&);

    void addQuad(const RenderLayerModelObject* repaintContainer, const FloatQuad&);
    void addGapRects(const RenderLayerModelObject* repaintContainer, const GapRects&);
    void setTextOnly(bool isTextOnly) { m_isTextOnly = isTextOnly; }
    bool isTextOnly() const { return m_isTextOnly; }

    class Notifier {
        WTF_MAKE_TZONE_ALLOCATED(Notifier);
        WTF_MAKE_NONCOPYABLE(Notifier);
    public:
        Notifier(SelectionGeometryGatherer&);
        ~Notifier();

    private:
        SelectionGeometryGatherer& m_gatherer;
    };

    std::unique_ptr<Notifier> clearAndCreateNotifier();
    
private:
    Vector<LayoutRect> boundingRects() const;

    SingleThreadWeakRef<RenderView> m_renderView;

    // All rects are in RenderView coordinates.
    Vector<FloatQuad> m_quads;
    Vector<GapRects> m_gapRects;
    bool m_isTextOnly;
};

} // namespace WebCore

#endif // ENABLE(SERVICE_CONTROLS)
