/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 28, 2021.
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

#include "HTMLInputElement.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class AutofillElements {
    WTF_MAKE_TZONE_ALLOCATED(AutofillElements);
public:
    WEBCORE_EXPORT static std::optional<AutofillElements> computeAutofillElements(Ref<HTMLInputElement>);
    WEBCORE_EXPORT void autofill(String, String);

    const HTMLInputElement* username() const { return m_username.get(); }
private:
    AutofillElements(RefPtr<HTMLInputElement>&&, RefPtr<HTMLInputElement>&&, RefPtr<HTMLInputElement>&&);
    RefPtr<HTMLInputElement> m_username;
    RefPtr<HTMLInputElement> m_password;
    RefPtr<HTMLInputElement> m_secondPassword;
};

} // namespace WebCore
