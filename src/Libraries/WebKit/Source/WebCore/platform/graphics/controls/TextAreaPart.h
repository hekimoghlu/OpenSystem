/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 29, 2023.
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

class TextAreaPart final : public ControlPart {
public:
    static Ref<TextAreaPart> create(StyleAppearance type)
    {
        return adoptRef(*new TextAreaPart(type));
    }

private:
    TextAreaPart(StyleAppearance type)
        : ControlPart(type)
    {
        ASSERT(type == StyleAppearance::Listbox || type == StyleAppearance::TextArea);
    }

    std::unique_ptr<PlatformControl> createPlatformControl() final
    {
        return controlFactory().createPlatformTextArea(*this);
    }
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::TextAreaPart) \
    static bool isType(const WebCore::ControlPart& part) { return part.type() == WebCore::StyleAppearance::Listbox || part.type() == WebCore::StyleAppearance::TextArea; } \
SPECIALIZE_TYPE_TRAITS_END()
