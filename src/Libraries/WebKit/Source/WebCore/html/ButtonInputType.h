/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 21, 2025.
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

#include "BaseButtonInputType.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class ButtonInputType final : public BaseButtonInputType {
    WTF_MAKE_TZONE_ALLOCATED(ButtonInputType);
public:
    static Ref<ButtonInputType> create(HTMLInputElement& element)
    {
        return adoptRef(*new ButtonInputType(element));
    }

private:
    explicit ButtonInputType(HTMLInputElement& element)
        : BaseButtonInputType(Type::Button, element)
    {
    }

    const AtomString& formControlType() const final;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_INPUT_TYPE(ButtonInputType, Type::Button)
