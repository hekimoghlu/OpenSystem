/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 13, 2023.
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
#include "HiddenInputType.h"

#include "DOMFormData.h"
#include "ElementInlines.h"
#include "FormController.h"
#include "HTMLInputElement.h"
#include "HTMLNames.h"
#include "InputTypeNames.h"
#include "RenderElement.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(HiddenInputType);

using namespace HTMLNames;

const AtomString& HiddenInputType::formControlType() const
{
    return InputTypeNames::hidden();
}

FormControlState HiddenInputType::saveFormControlState() const
{
    // valueAttributeWasUpdatedAfterParsing() never be true for form controls create by createElement() or cloneNode().
    // It's OK for now because we restore values only to form controls created by parsing.
    ASSERT(element());
    return element()->valueAttributeWasUpdatedAfterParsing() ? FormControlState { { AtomString { element()->value() } } } : FormControlState { };
}

void HiddenInputType::restoreFormControlState(const FormControlState& state)
{
    ASSERT(element());
    element()->setAttributeWithoutSynchronization(valueAttr, AtomString { state[0] });
}

RenderPtr<RenderElement> HiddenInputType::createInputRenderer(RenderStyle&&)
{
    ASSERT_NOT_REACHED();
    return nullptr;
}

bool HiddenInputType::accessKeyAction(bool)
{
    return false;
}

bool HiddenInputType::rendererIsNeeded()
{
    return false;
}

bool HiddenInputType::storesValueSeparateFromAttribute()
{
    return false;
}

void HiddenInputType::setValue(const String& sanitizedValue, bool, TextFieldEventBehavior, TextControlSetValueSelection)
{
    ASSERT(element());
    element()->setAttributeWithoutSynchronization(valueAttr, AtomString { sanitizedValue });
}

bool HiddenInputType::appendFormData(DOMFormData& formData) const
{
    ASSERT(element());
    auto name = element()->name();

    if (equalIgnoringASCIICase(name, "_charset_"_s)) {
        formData.append(name, String::fromLatin1(formData.encoding().name()));
        return true;
    }
    InputType::appendFormData(formData);
    if (auto& dirname = element()->attributeWithoutSynchronization(dirnameAttr); !dirname.isNull())
        formData.append(dirname, element()->directionForFormData());
    return true;
}

bool HiddenInputType::shouldRespectHeightAndWidthAttributes()
{
    return true;
}

bool HiddenInputType::dirAutoUsesValue() const
{
    return true;
}

} // namespace WebCore
