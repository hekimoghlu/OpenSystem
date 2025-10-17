/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 5, 2022.
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

#include "CachedFontClient.h"
#include "CachedResourceHandle.h"
#include "SVGElement.h"

namespace WebCore {

class CSSFontFaceSrcResourceValue;

class SVGFontFaceUriElement final : public SVGElement, public CachedFontClient {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(SVGFontFaceUriElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(SVGFontFaceUriElement);
public:
    static Ref<SVGFontFaceUriElement> create(const QualifiedName&, Document&);

    virtual ~SVGFontFaceUriElement();

    Ref<CSSFontFaceSrcResourceValue> createSrcValue() const;

private:
    SVGFontFaceUriElement(const QualifiedName&, Document&);

    void attributeChanged(const QualifiedName&, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason) final;
    void childrenChanged(const ChildChange&) final;
    InsertedIntoAncestorResult insertedIntoAncestor(InsertionType, ContainerNode&) final;
    bool rendererIsNeeded(const RenderStyle&) final { return false; }

    void loadFont();

    CachedResourceHandle<CachedFont> m_cachedFont;
};

} // namespace WebCore
