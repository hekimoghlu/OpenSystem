/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 4, 2023.
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

#include "HTMLFrameElement.h"
#include "RenderFrameBase.h"

namespace WebCore {

struct FrameEdgeInfo;

class RenderFrame final : public RenderFrameBase {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(RenderFrame);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(RenderFrame);
public:
    RenderFrame(HTMLFrameElement&, RenderStyle&&);
    virtual ~RenderFrame();

    HTMLFrameElement& frameElement() const;
    FrameEdgeInfo edgeInfo() const;

    void updateFromElement() final;

private:
    void frameOwnerElement() const = delete;

    ASCIILiteral renderName() const final { return "RenderFrame"_s; }
};

inline RenderFrame* HTMLFrameElement::renderer() const
{
    return downcast<RenderFrame>(HTMLFrameElementBase::renderer());
}

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_RENDER_OBJECT(RenderFrame, isRenderFrame())
