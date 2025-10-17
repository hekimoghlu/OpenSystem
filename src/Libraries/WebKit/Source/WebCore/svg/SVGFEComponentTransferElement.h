/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 11, 2022.
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

#include "SVGFilterPrimitiveStandardAttributes.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class SVGComponentTransferFunctionElement;

class SVGFEComponentTransferElement final : public SVGFilterPrimitiveStandardAttributes {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(SVGFEComponentTransferElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(SVGFEComponentTransferElement);
public:
    static Ref<SVGFEComponentTransferElement> create(const QualifiedName&, Document&);

    String in1() const { return m_in1->currentValue(); }
    SVGAnimatedString& in1Animated() { return m_in1; }

    void transferFunctionAttributeChanged(SVGComponentTransferFunctionElement&, const QualifiedName&);

protected:
    bool setFilterEffectAttributeFromChild(FilterEffect&, const Element&, const QualifiedName&) final;

    using PropertyRegistry = SVGPropertyOwnerRegistry<SVGFEComponentTransferElement, SVGFilterPrimitiveStandardAttributes>;

private:
    SVGFEComponentTransferElement(const QualifiedName&, Document&);

    void attributeChanged(const QualifiedName&, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason) override;
    void svgAttributeChanged(const QualifiedName&) override;

    Vector<AtomString> filterEffectInputsNames() const override { return { AtomString { in1() } }; }
    RefPtr<FilterEffect> createFilterEffect(const FilterEffectVector&, const GraphicsContext& destinationContext) const override;

    Ref<SVGAnimatedString> m_in1 { SVGAnimatedString::create(this) };
};

} // namespace WebCore
