/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 28, 2025.
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

#include "SVGAnimateElementBase.h"
#include "SVGTransformValue.h"

namespace WebCore {

class AffineTransform;

class SVGAnimateTransformElement final : public SVGAnimateElementBase {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(SVGAnimateTransformElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(SVGAnimateTransformElement);
public:
    static Ref<SVGAnimateTransformElement> create(const QualifiedName&, Document&);

    SVGTransformValue::SVGTransformType transformType() const { return m_type; }

private:
    SVGAnimateTransformElement(const QualifiedName&, Document&);
    
    bool hasValidAttributeType() const final;
    void attributeChanged(const QualifiedName&, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason) final;
    String animateRangeString(const String&) const final;

    SVGTransformValue::SVGTransformType m_type;
};

} // namespace WebCore
