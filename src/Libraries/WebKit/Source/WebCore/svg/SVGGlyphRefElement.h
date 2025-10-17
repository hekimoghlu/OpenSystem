/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 7, 2024.
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
#include "SVGURIReference.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class SVGGlyphRefElement final : public SVGElement, public SVGURIReference {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(SVGGlyphRefElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(SVGGlyphRefElement);
public:
    static Ref<SVGGlyphRefElement> create(const QualifiedName&, Document&);

    bool hasValidGlyphElement(String& glyphName) const;

    float x() const { return m_x; }
    void setX(float);
    float y() const { return m_y; }
    void setY(float);
    float dx() const { return m_dx; }
    void setDx(float);
    float dy() const { return m_dy; }
    void setDy(float);

    using PropertyRegistry = SVGPropertyOwnerRegistry<SVGGlyphRefElement, SVGElement, SVGURIReference>;

private:
    SVGGlyphRefElement(const QualifiedName&, Document&);

    void attributeChanged(const QualifiedName&, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason) final;
    bool rendererIsNeeded(const RenderStyle&) final { return false; }

    float m_x { 0 };
    float m_y { 0 };
    float m_dx { 0 };
    float m_dy { 0 };
};

}
