/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 27, 2023.
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

// This type is used for 2 kinds of elements:
// - Unknown Elements in SVG namespace
// - Registered custom tag elements in SVG namespace (http://www.w3.org/TR/2013/WD-custom-elements-20130514/#registering-custom-elements)
//
// The main purpose of this class at the moment is to override rendererIsNeeded() to return
// false to make sure we don't attempt to render such elements.
class SVGUnknownElement final : public SVGElement {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(SVGUnknownElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(SVGUnknownElement);
public:
    static Ref<SVGUnknownElement> create(const QualifiedName& tagName, Document& document)
    {
        return adoptRef(*new SVGUnknownElement(tagName, document));
    }

private:
    SVGUnknownElement(const QualifiedName& tagName, Document& document)
        : SVGElement(tagName, document, makeUniqueRef<PropertyRegistry>(*this), TypeFlag::IsUnknownElement)
    {
    }

    bool rendererIsNeeded(const RenderStyle&) final { return false; }
};

} // namespace WebCore
