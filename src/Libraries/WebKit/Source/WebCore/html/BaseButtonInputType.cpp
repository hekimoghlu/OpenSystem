/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 7, 2024.
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
#include "config.h"
#include "BaseButtonInputType.h"

#include "HTMLInputElement.h"
#include "HTMLNames.h"
#include "RenderButton.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(BaseButtonInputType);

using namespace HTMLNames;

bool BaseButtonInputType::shouldSaveAndRestoreFormControlState() const
{
    return false;
}

bool BaseButtonInputType::appendFormData(DOMFormData&) const
{
    // Buttons except overridden types are never successful.
    return false;
}

RenderPtr<RenderElement> BaseButtonInputType::createInputRenderer(RenderStyle&& style)
{
    ASSERT(element());
    return createRenderer<RenderButton>(*element(), WTFMove(style));
}

bool BaseButtonInputType::storesValueSeparateFromAttribute()
{
    return false;
}

void BaseButtonInputType::setValue(const String& sanitizedValue, bool, TextFieldEventBehavior, TextControlSetValueSelection)
{
    ASSERT(element());
    element()->setAttributeWithoutSynchronization(valueAttr, AtomString { sanitizedValue });
}

bool BaseButtonInputType::dirAutoUsesValue() const
{
    return true;
}

} // namespace WebCore
