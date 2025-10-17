/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 16, 2024.
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

#if ENABLE(MODEL_ELEMENT)

#include "RenderReplaced.h"

namespace WebCore {

class HTMLModelElement;

class RenderModel final : public RenderReplaced {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(RenderModel);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(RenderModel);
public:
    RenderModel(HTMLModelElement&, RenderStyle&&);
    virtual ~RenderModel();

    HTMLModelElement& modelElement() const;

private:
    void element() const = delete;
    ASCIILiteral renderName() const final { return "RenderModel"_s; }

    bool requiresLayer() const final;
    void updateFromElement() final;

    void update();
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_RENDER_OBJECT(RenderModel, isRenderModel())

#endif
