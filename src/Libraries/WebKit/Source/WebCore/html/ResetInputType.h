/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 17, 2022.
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

class ResetInputType final : public BaseButtonInputType {
    WTF_MAKE_TZONE_ALLOCATED(ResetInputType);
public:
    static Ref<ResetInputType> create(HTMLInputElement& element)
    {
        return adoptRef(*new ResetInputType(element));
    }

private:
    explicit ResetInputType(HTMLInputElement& element)
        : BaseButtonInputType(Type::Reset, element)
    {
    }

    const AtomString& formControlType() const final;
    void handleDOMActivateEvent(Event&) final;
    String defaultValue() const final;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_INPUT_TYPE(ResetInputType, Type::Reset)
