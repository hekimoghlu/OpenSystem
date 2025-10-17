/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 23, 2022.
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

class EmailInputType final : public BaseTextInputType {
    WTF_MAKE_TZONE_ALLOCATED(EmailInputType);
public:
    static Ref<EmailInputType> create(HTMLInputElement& element)
    {
        return adoptRef(*new EmailInputType(element));
    }

    bool typeMismatchFor(const String&) const final;
    bool typeMismatch() const final;

private:
    explicit EmailInputType(HTMLInputElement& element)
        : BaseTextInputType(Type::Email, element)
    {
    }

    const AtomString& formControlType() const final;
    String typeMismatchText() const final;
    bool supportsSelectionAPI() const final;
    String sanitizeValue(const String&) const final;
    void attributeChanged(const QualifiedName&) final;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_INPUT_TYPE(EmailInputType, Type::Email)
