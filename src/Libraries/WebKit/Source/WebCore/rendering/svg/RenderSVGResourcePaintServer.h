/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 6, 2021.
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

#include "Color.h"
#include "RenderSVGResourceContainer.h"
#include <variant>

namespace WebCore {

class GraphicsContext;
class RenderSVGShape;

class RenderSVGResourcePaintServer : public RenderSVGResourceContainer {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(RenderSVGResourcePaintServer);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(RenderSVGResourcePaintServer);
public:
    virtual ~RenderSVGResourcePaintServer();

    virtual bool prepareFillOperation(GraphicsContext&, const RenderLayerModelObject&, const RenderStyle&) { return false; }
    virtual bool prepareStrokeOperation(GraphicsContext&, const RenderLayerModelObject&, const RenderStyle&) { return false; }

protected:
    RenderSVGResourcePaintServer(Type, SVGElement&, RenderStyle&&);
};

using SVGPaintServerOrColor = std::variant<std::monostate, RenderSVGResourcePaintServer*, Color>;

}

SPECIALIZE_TYPE_TRAITS_RENDER_OBJECT(RenderSVGResourcePaintServer, isRenderSVGResourcePaintServer())
