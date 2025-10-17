/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 21, 2025.
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

#include "RenderInline.h"

namespace WebCore {

class RenderQuote final : public RenderInline {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(RenderQuote);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(RenderQuote);
public:
    RenderQuote(Document&, RenderStyle&&, QuoteType);
    virtual ~RenderQuote();

    void updateRenderer(RenderTreeBuilder&, RenderQuote* previousQuote);

private:
    ASCIILiteral renderName() const override { return "RenderQuote"_s; }
    bool isOpen() const;
    void styleDidChange(StyleDifference, const RenderStyle*) override;
    void insertedIntoTree() override;
    void willBeRemovedFromTree() override;

    String computeText() const;
    void updateTextRenderer(RenderTreeBuilder&);

    const QuoteType m_type;
    int m_depth { -1 };
    String m_text;

    bool m_needsTextUpdate { false };
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_RENDER_OBJECT(RenderQuote, isRenderQuote())
