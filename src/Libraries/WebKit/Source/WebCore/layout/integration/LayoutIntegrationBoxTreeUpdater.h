/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 16, 2025.
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

#include <wtf/CheckedRef.h>
#include <wtf/UniqueRef.h>

namespace WebCore {

class RenderBlock;
class RenderBoxModelObject;
class RenderElement;
class RenderObject;
class RenderInline;
class RenderStyle;
class RenderText;

namespace Layout {
class Box;
class ElementBox;
class InitialContainingBlock;
}

namespace LayoutIntegration {

#if ENABLE(TREE_DEBUGGING)
struct InlineContent;
#endif

class BoxTreeUpdater {
public:
    BoxTreeUpdater(RenderBlock&);
    ~BoxTreeUpdater();

    CheckedRef<Layout::ElementBox> build();
    void tearDown();

    static void updateStyle(const RenderObject&);
    static void updateContent(const RenderText&);

    const Layout::Box& insert(const RenderElement& parent, RenderObject& child, const RenderObject* beforeChild = nullptr);
    UniqueRef<Layout::Box> remove(const RenderElement& parent, RenderObject& child);

private:
    const RenderBlock& rootRenderer() const { return m_rootRenderer; }
    RenderBlock& rootRenderer() { return m_rootRenderer; }

    const Layout::ElementBox& rootLayoutBox() const;
    Layout::ElementBox& rootLayoutBox();

    Layout::InitialContainingBlock& initialContainingBlock();

    static UniqueRef<Layout::Box> createLayoutBox(RenderObject&);
    static void adjustStyleIfNeeded(const RenderElement&, RenderStyle&, RenderStyle* firstLineStyle);

    void buildTreeForInlineContent();
    void buildTreeForFlexContent();
    void insertChild(UniqueRef<Layout::Box>, RenderObject&, const RenderObject* beforeChild = nullptr);

    RenderBlock& m_rootRenderer;
};

#if ENABLE(TREE_DEBUGGING)
void showInlineContent(TextStream&, const InlineContent&, size_t depth, bool isDamaged = false);
#endif
}
}

