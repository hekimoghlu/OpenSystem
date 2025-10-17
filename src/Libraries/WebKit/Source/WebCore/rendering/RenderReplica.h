/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 1, 2024.
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

#include "RenderBox.h"

namespace WebCore {

class RenderReplica final : public RenderBox {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(RenderReplica);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(RenderReplica);
public:
    RenderReplica(Document&, RenderStyle&&);
    virtual ~RenderReplica();

    ASCIILiteral renderName() const override { return "RenderReplica"_s; }

    bool requiresLayer() const override { return true; }

    void layout() override;
    
    void paint(PaintInfo&, const LayoutPoint&) override;

private:
    bool canHaveChildren() const override { return false; }
    void computePreferredLogicalWidths() override;
    void computeIntrinsicLogicalWidths(LayoutUnit&, LayoutUnit&) const override { ASSERT_NOT_REACHED(); }
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_RENDER_OBJECT(RenderReplica, isRenderReplica())
