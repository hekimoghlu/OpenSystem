/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 29, 2023.
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

#include "BaseTextInputType.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class PasswordInputType final : public BaseTextInputType {
    WTF_MAKE_TZONE_ALLOCATED(PasswordInputType);
public:
    static Ref<PasswordInputType> create(HTMLInputElement& element)
    {
        return adoptRef(*new PasswordInputType(element));
    }

private:
    explicit PasswordInputType(HTMLInputElement& element)
        : BaseTextInputType(Type::Password, element)
    {
    }

    const AtomString& formControlType() const final;
    bool shouldSaveAndRestoreFormControlState() const final;
    FormControlState saveFormControlState() const final;
    void restoreFormControlState(const FormControlState&) final;
    bool shouldUseInputMethod() const final;
    bool shouldResetOnDocumentActivation() final;
    bool shouldRespectListAttribute() final;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_INPUT_TYPE(PasswordInputType, Type::Password)
