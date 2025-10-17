/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 29, 2022.
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

#include "LayoutElementBox.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {
namespace Layout {

class InitialContainingBlock final : public ElementBox {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(InitialContainingBlock);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(InitialContainingBlock);
public:
    InitialContainingBlock(RenderStyle&&, std::unique_ptr<RenderStyle>&& firstLineStyle = nullptr);
    virtual ~InitialContainingBlock() = default;

private:
    const ElementBox& parent() const = delete;
    const Box* nextSibling() const = delete;
    const Box* nextInFlowSibling() const = delete;
    const Box* nextInFlowOrFloatingSibling() const = delete;
    const Box* previousSibling() const = delete;
    const Box* previousInFlowSibling() const = delete;
    const Box* previousInFlowOrFloatingSibling() const = delete;
    Box* nextSibling() = delete;
};

}
}

SPECIALIZE_TYPE_TRAITS_LAYOUT_BOX(InitialContainingBlock, isInitialContainingBlock())

