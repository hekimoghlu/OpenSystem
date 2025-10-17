/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 30, 2021.
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

#include "RenderSVGHiddenContainer.h"

namespace WebCore {

class RenderSVGResourceContainer : public RenderSVGHiddenContainer {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(RenderSVGResourceContainer);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(RenderSVGResourceContainer);
public:
    virtual ~RenderSVGResourceContainer();

    void styleDidChange(StyleDifference, const RenderStyle* oldStyle) override;

    void idChanged();
    void repaintAllClients() const;

    virtual void addReferencingCSSClient(const RenderElement&) { }
    virtual void removeReferencingCSSClient(const RenderElement&) { }

protected:
    RenderSVGResourceContainer(Type, SVGElement&, RenderStyle&&);

private:
    void willBeDestroyed() final;
    void registerResource();

    AtomString m_id;
    bool m_registered { false };
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_RENDER_OBJECT(RenderSVGResourceContainer, isRenderSVGResourceContainer())

