/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 18, 2024.
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

#include "SVGGraphicsElement.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class SVGDefsElement final : public SVGGraphicsElement {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(SVGDefsElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(SVGDefsElement);
public:
    static Ref<SVGDefsElement> create(const QualifiedName&, Document&);

    using PropertyRegistry = SVGPropertyOwnerRegistry<SVGDefsElement, SVGGraphicsElement>;

private:
    SVGDefsElement(const QualifiedName&, Document&);

    bool isValid() const final;
    bool supportsFocus() const final { return false; }

    RenderPtr<RenderElement> createElementRenderer(RenderStyle&&, const RenderTreePosition&) final;
};

} // namespace WebCore
