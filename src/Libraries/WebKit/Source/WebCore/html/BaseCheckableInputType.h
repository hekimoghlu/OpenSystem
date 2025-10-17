/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 26, 2023.
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

// Base of checkbox and radio types.
class BaseCheckableInputType : public InputType {
    WTF_MAKE_TZONE_ALLOCATED(BaseCheckableInputType);
public:
    bool canSetStringValue() const final;

protected:
    explicit BaseCheckableInputType(Type type, HTMLInputElement& element)
        : InputType(type, element)
    {
    }

    ShouldCallBaseEventHandler handleKeydownEvent(KeyboardEvent&) override;
    void fireInputAndChangeEvents();

private:
    FormControlState saveFormControlState() const final;
    void restoreFormControlState(const FormControlState&) final;
    bool appendFormData(DOMFormData&) const final;
    void handleKeypressEvent(KeyboardEvent&) final;
    bool accessKeyAction(bool sendMouseEvents) final;
    String fallbackValue() const final;
    bool storesValueSeparateFromAttribute() final;
    void setValue(const String&, bool, TextFieldEventBehavior, TextControlSetValueSelection) final;
};

} // namespace WebCore
