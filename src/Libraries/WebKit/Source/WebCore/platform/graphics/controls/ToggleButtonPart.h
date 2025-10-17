/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 8, 2022.
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

#include "ControlFactory.h"
#include "ControlPart.h"

namespace WebCore {

class ToggleButtonPart final : public ControlPart {
public:
    static Ref<ToggleButtonPart> create(StyleAppearance type)
    {
        return adoptRef(*new ToggleButtonPart(type));
    }

private:
    ToggleButtonPart(StyleAppearance type)
        : ControlPart(type)
    {
        ASSERT(type == StyleAppearance::Checkbox || type == StyleAppearance::Radio);
    }

    std::unique_ptr<PlatformControl> createPlatformControl() final
    {
        return controlFactory().createPlatformToggleButton(*this);
    }
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::ToggleButtonPart) \
    static bool isType(const WebCore::ControlPart& part) { return part.type() == WebCore::StyleAppearance::Checkbox || part.type() == WebCore::StyleAppearance::Radio; } \
SPECIALIZE_TYPE_TRAITS_END()
