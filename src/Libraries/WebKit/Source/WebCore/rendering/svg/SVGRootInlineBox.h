/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 22, 2024.
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

#include "LegacyRootInlineBox.h"
#include "SVGTextLayoutEngine.h"

namespace WebCore {

class RenderSVGText;
class SVGInlineTextBox;

class SVGRootInlineBox final : public LegacyRootInlineBox {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(SVGRootInlineBox);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(SVGRootInlineBox);
public:
    explicit SVGRootInlineBox(RenderSVGText&);

    float virtualLogicalHeight() const override { return m_logicalHeight; }
    void setLogicalHeight(float height) { m_logicalHeight = height; }

private:
    RenderSVGText& renderSVGText() const;

    bool isSVGRootInlineBox() const override { return true; }

    float m_logicalHeight;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_INLINE_BOX(SVGRootInlineBox, isSVGRootInlineBox())
