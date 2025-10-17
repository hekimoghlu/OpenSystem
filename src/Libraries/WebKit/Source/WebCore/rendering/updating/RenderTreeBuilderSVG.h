/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 2, 2021.
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

#include "RenderTreeBuilder.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class LegacyRenderSVGContainer;
class LegacyRenderSVGRoot;
class RenderSVGContainer;
class RenderSVGViewportContainer;
class RenderSVGInline;
class RenderSVGRoot;
class RenderSVGText;

class RenderTreeBuilder::SVG {
    WTF_MAKE_TZONE_ALLOCATED(SVG);
public:
    SVG(RenderTreeBuilder&);

    void updateAfterDescendants(RenderSVGRoot&);

    void attach(LegacyRenderSVGRoot& parent, RenderPtr<RenderObject> child, RenderObject* beforeChild);
    void attach(LegacyRenderSVGContainer& parent, RenderPtr<RenderObject> child, RenderObject* beforeChild);
    void attach(RenderSVGInline& parent, RenderPtr<RenderObject> child, RenderObject* beforeChild);
    void attach(RenderSVGText& parent, RenderPtr<RenderObject> child, RenderObject* beforeChild);
    void attach(RenderSVGRoot& parent, RenderPtr<RenderObject> child, RenderObject* beforeChild);

    RenderPtr<RenderObject> detach(LegacyRenderSVGRoot& parent, RenderObject& child, RenderTreeBuilder::WillBeDestroyed) WARN_UNUSED_RETURN;
    RenderPtr<RenderObject> detach(LegacyRenderSVGContainer& parent, RenderObject& child, RenderTreeBuilder::WillBeDestroyed) WARN_UNUSED_RETURN;
    RenderPtr<RenderObject> detach(RenderSVGInline& parent, RenderObject& child, RenderTreeBuilder::WillBeDestroyed) WARN_UNUSED_RETURN;

    RenderPtr<RenderObject> detach(RenderSVGText& parent, RenderObject& child, RenderTreeBuilder::WillBeDestroyed) WARN_UNUSED_RETURN;

private:
    RenderSVGViewportContainer& findOrCreateParentForChild(RenderSVGRoot&);
    RenderSVGViewportContainer& createViewportContainer(RenderSVGRoot&);
    RenderTreeBuilder& m_builder;
};

}
