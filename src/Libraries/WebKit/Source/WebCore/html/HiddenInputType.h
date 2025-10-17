/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 18, 2022.
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

#include "InputType.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class HiddenInputType final : public InputType {
    WTF_MAKE_TZONE_ALLOCATED(HiddenInputType);
public:
    static Ref<HiddenInputType> create(HTMLInputElement& element)
    {
        return adoptRef(*new HiddenInputType(element));
    }

private:
    explicit HiddenInputType(HTMLInputElement& element)
        : InputType(Type::Hidden, element)
    {
    }

    const AtomString& formControlType() const final;
    FormControlState saveFormControlState() const final;
    void restoreFormControlState(const FormControlState&) final;
    RenderPtr<RenderElement> createInputRenderer(RenderStyle&&) final;
    bool accessKeyAction(bool sendMouseEvents) final;
    bool rendererIsNeeded() final;
    bool storesValueSeparateFromAttribute() final;
    bool shouldRespectHeightAndWidthAttributes() final;
    void setValue(const String&, bool, TextFieldEventBehavior, TextControlSetValueSelection) final;
    bool appendFormData(DOMFormData&) const final;
    bool dirAutoUsesValue() const final;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_INPUT_TYPE(HiddenInputType, Type::Hidden)
