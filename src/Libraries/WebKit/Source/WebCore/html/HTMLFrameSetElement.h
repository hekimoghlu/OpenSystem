/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 17, 2022.
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

#include "HTMLElement.h"
#include <wtf/UniqueArray.h>

namespace WebCore {

class WindowProxy;

class HTMLFrameSetElement final : public HTMLElement {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(HTMLFrameSetElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(HTMLFrameSetElement);
public:
    static Ref<HTMLFrameSetElement> create(const QualifiedName&, Document&);

    bool hasFrameBorder() const { return m_frameborder; }
    bool noResize() const { return m_noresize; }

    int totalRows() const { return m_totalRows; }
    int totalCols() const { return m_totalCols; }
    int border() const { return hasFrameBorder() ? m_border : 0; }

    bool hasBorderColor() const { return m_borderColorSet; }

    std::span<const Length> rowLengths() const { return m_rowLengths ? unsafeMakeSpan(m_rowLengths.get(), m_totalRows) : std::span<const Length> { }; }
    std::span<const Length> colLengths() const { return m_colLengths ? unsafeMakeSpan(m_colLengths.get(), m_totalCols) : std::span<const Length> { }; }

    static RefPtr<HTMLFrameSetElement> findContaining(Element* descendant);
    
    Vector<AtomString> supportedPropertyNames() const;
    WindowProxy* namedItem(const AtomString&);
    bool isSupportedPropertyName(const AtomString&);

private:
    HTMLFrameSetElement(const QualifiedName&, Document&);

    void attributeChanged(const QualifiedName&, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason) final;
    bool hasPresentationalHintsForAttribute(const QualifiedName&) const final;
    void collectPresentationalHintsForAttribute(const QualifiedName&, const AtomString&, MutableStyleProperties&) final;

    void willAttachRenderers() final;
    RenderPtr<RenderElement> createElementRenderer(RenderStyle&&, const RenderTreePosition&) final;
    
    void defaultEventHandler(Event&) final;

    void willRecalcStyle(Style::Change) final;

    InsertedIntoAncestorResult insertedIntoAncestor(InsertionType, ContainerNode&) final;
    void removedFromAncestor(RemovalType, ContainerNode&) final;

    UniqueArray<Length> m_rowLengths;
    UniqueArray<Length> m_colLengths;

    int m_totalRows;
    int m_totalCols;
    
    int m_border;
    bool m_borderSet;
    
    bool m_borderColorSet;

    bool m_frameborder;
    bool m_frameborderSet;
    bool m_noresize;
};

} // namespace WebCore
