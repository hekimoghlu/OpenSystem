/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 23, 2022.
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

#include "RenderFrameBase.h"

namespace WebCore {

class RenderView;

class RenderIFrame final : public RenderFrameBase {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(RenderIFrame);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(RenderIFrame);
public:
    RenderIFrame(HTMLIFrameElement&, RenderStyle&&);
    virtual ~RenderIFrame();

    HTMLIFrameElement& iframeElement() const;

private:
    void frameOwnerElement() const = delete;

    bool isNonReplacedAtomicInline() const override;

    void layout() override;

    ASCIILiteral renderName() const override { return "RenderIFrame"_s; }

    bool requiresLayer() const override;

    bool isFullScreenIFrame() const;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_RENDER_OBJECT(RenderIFrame, isRenderIFrame())
