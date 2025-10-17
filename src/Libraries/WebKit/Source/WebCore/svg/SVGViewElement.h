/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 31, 2025.
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
#include "SVGFitToViewBox.h"
#include "SVGSVGElement.h"
#include "SVGStringList.h"
#include "SVGZoomAndPan.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class SVGViewElement final : public SVGElement, public SVGFitToViewBox, public SVGZoomAndPan {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(SVGViewElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(SVGViewElement);
public:
    static Ref<SVGViewElement> create(const QualifiedName&, Document&);

    using PropertyRegistry = SVGPropertyOwnerRegistry<SVGViewElement, SVGElement, SVGFitToViewBox>;
    using SVGElement::ref;
    using SVGElement::deref;

    const SVGSVGElement* targetElement() const { return m_targetElement.get(); }
    void setTargetElement(const SVGSVGElement& targetElement) { m_targetElement = targetElement; }
    void resetTargetElement() { m_targetElement = nullptr; }

private:
    SVGViewElement(const QualifiedName&, Document&);

    void attributeChanged(const QualifiedName&, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason) final;
    void svgAttributeChanged(const QualifiedName&) override;

    bool rendererIsNeeded(const RenderStyle&) final { return false; }

    WeakPtr<SVGSVGElement, WeakPtrImplWithEventTargetData> m_targetElement;
};

} // namespace WebCore
