/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 22, 2024.
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

#include "RenderBlockFlow.h"

namespace WebCore {

class RenderTable;

class RenderTableCaption final : public RenderBlockFlow {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(RenderTableCaption);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(RenderTableCaption);
public:
    RenderTableCaption(Element&, RenderStyle&&);
    virtual ~RenderTableCaption();

    Element& element() const { return downcast<Element>(nodeForNonAnonymous()); }

private:
    void insertedIntoTree() override;
    void willBeRemovedFromTree() override;
    LayoutUnit containingBlockLogicalWidthForContent() const final;

    RenderTable* table() const;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_RENDER_OBJECT(RenderTableCaption, isRenderTableCaption())
