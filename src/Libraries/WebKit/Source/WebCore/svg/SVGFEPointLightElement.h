/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 3, 2021.
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

#include "SVGFELightElement.h"

namespace WebCore {

class SVGFEPointLightElement final : public SVGFELightElement {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(SVGFEPointLightElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(SVGFEPointLightElement);
public:
    static Ref<SVGFEPointLightElement> create(const QualifiedName&, Document&);

private:
    SVGFEPointLightElement(const QualifiedName&, Document&);

    Ref<LightSource> lightSource() const override;
};

static_assert(sizeof(SVGFEPointLightElement) == sizeof(SVGFELightElement));

} // namespace WebCore
