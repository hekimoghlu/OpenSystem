/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 26, 2022.
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

#include "HTMLDivElement.h"

namespace WebCore {

class TextFieldInputType;

class AutoFillButtonElement final : public HTMLDivElement {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(AutoFillButtonElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(AutoFillButtonElement);
public:
    class AutoFillButtonOwner {
    public:
        virtual ~AutoFillButtonOwner() = default;
        virtual void autoFillButtonElementWasClicked() = 0;
    };

    static Ref<AutoFillButtonElement> create(Document&, AutoFillButtonOwner&);

private:
    explicit AutoFillButtonElement(Document&, AutoFillButtonOwner&);

    void defaultEventHandler(Event&) override;

    AutoFillButtonOwner& m_owner;
};

} // namespace WebCore
