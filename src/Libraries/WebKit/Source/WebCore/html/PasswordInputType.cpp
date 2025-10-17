/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 15, 2023.
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
#include "PasswordInputType.h"

#include "FormController.h"
#include "HTMLInputElement.h"
#include "InputTypeNames.h"
#include <wtf/Assertions.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(PasswordInputType);

const AtomString& PasswordInputType::formControlType() const
{
    return InputTypeNames::password();
}

bool PasswordInputType::shouldSaveAndRestoreFormControlState() const
{
    return false;
}

FormControlState PasswordInputType::saveFormControlState() const
{
    // Should never save/restore password fields.
    ASSERT_NOT_REACHED();
    return FormControlState();
}

void PasswordInputType::restoreFormControlState(const FormControlState&)
{
    // Should never save/restore password fields.
    ASSERT_NOT_REACHED();
}

bool PasswordInputType::shouldUseInputMethod() const
{
#if PLATFORM(GTK) || PLATFORM(WPE)
    // Input methods are enabled for the password field in GTK and WPE ports
    // because the input methods are notified that the active editable element
    // is a password field.
    return true;
#else
    // Input methods are disabled for the password field because otherwise
    // anyone can access the underlying password and display it in clear text.
    return false;
#endif
}

bool PasswordInputType::shouldResetOnDocumentActivation()
{
    return true;
}

bool PasswordInputType::shouldRespectListAttribute()
{
    return false;
}

} // namespace WebCore
