/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 7, 2022.
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

#include "SVGElement.h"

namespace WebCore {

class SVGFontElement;
class StyleRuleFontFace;

class SVGFontFaceElement final : public SVGElement {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(SVGFontFaceElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(SVGFontFaceElement);
public:
    static Ref<SVGFontFaceElement> create(const QualifiedName&, Document&);

    unsigned unitsPerEm() const;
    int xHeight() const;
    int capHeight() const;
    float horizontalOriginX() const;
    float horizontalOriginY() const;
    float horizontalAdvanceX() const;
    float verticalOriginX() const;
    float verticalOriginY() const;
    float verticalAdvanceY() const;
    int ascent() const;
    int descent() const;
    String fontFamily() const;

    SVGFontElement* associatedFontElement() const;
    RefPtr<SVGFontElement> protectedFontElement() const;
    void rebuildFontFace();
    
    StyleRuleFontFace& fontFaceRule() { return m_fontFaceRule.get(); }
    Ref<StyleRuleFontFace> protectedFontFaceRule() const;

private:
    SVGFontFaceElement(const QualifiedName&, Document&);
    ~SVGFontFaceElement();

    void attributeChanged(const QualifiedName&, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason) final;

    void childrenChanged(const ChildChange&) final;
    InsertedIntoAncestorResult insertedIntoAncestor(InsertionType, ContainerNode&) final;
    void removedFromAncestor(RemovalType, ContainerNode&) final;

    bool rendererIsNeeded(const RenderStyle&) final { return false; }

    Ref<StyleRuleFontFace> m_fontFaceRule;
    WeakPtr<SVGFontElement, WeakPtrImplWithEventTargetData> m_fontElement;
};

} // namespace WebCore
