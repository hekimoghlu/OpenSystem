/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 14, 2023.
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

#include "RenderLayerModelObject.h"

#include <functional>
#include <wtf/Noncopyable.h>
#include <wtf/Vector.h>

namespace WebCore {

class SVGContainerLayout {
    WTF_MAKE_NONCOPYABLE(SVGContainerLayout);
public:
    explicit SVGContainerLayout(RenderLayerModelObject&);
    ~SVGContainerLayout() = default;

    // 'containerNeedsLayout' denotes if the container for which the
    // SVGContainerLayout object was created needs to be laid out or not.
    void layoutChildren(bool containerNeedsLayout);

    void positionChildrenRelativeToContainer();

    static void verifyLayoutLocationConsistency(const RenderLayerModelObject&);
    static bool transformToRootChanged(const RenderObject* ancestor);

private:
    bool layoutSizeOfNearestViewportChanged() const;

    SingleThreadWeakRef<RenderLayerModelObject> m_container;
    Vector<std::reference_wrapper<RenderLayerModelObject>> m_positionedChildren;
};

} // namespace WebCore

