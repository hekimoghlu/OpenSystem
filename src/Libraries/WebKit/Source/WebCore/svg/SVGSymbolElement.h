/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 30, 2022.
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

#include "SVGFitToViewBox.h"
#include "SVGGraphicsElement.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class SVGSymbolElement final : public SVGGraphicsElement, public SVGFitToViewBox {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(SVGSymbolElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(SVGSymbolElement);
public:
    static Ref<SVGSymbolElement> create(const QualifiedName&, Document&);

    using PropertyRegistry = SVGPropertyOwnerRegistry<SVGSymbolElement, SVGGraphicsElement, SVGFitToViewBox>;

private:
    SVGSymbolElement(const QualifiedName&, Document&);

    void attributeChanged(const QualifiedName&, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason) override;
    RenderPtr<RenderElement> createElementRenderer(RenderStyle&&, const RenderTreePosition&) override;

    bool selfHasRelativeLengths() const override;
    bool supportsFocus() const final { return false; }
};

} // namespace WebCore
