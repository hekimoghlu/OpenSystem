/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 16, 2024.
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

class DataListButtonElement final : public HTMLDivElement {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(DataListButtonElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(DataListButtonElement);
public:
    class DataListButtonOwner {
    public:
        virtual ~DataListButtonOwner() = default;
        virtual void dataListButtonElementWasClicked() = 0;
    };

    ~DataListButtonElement();

    static Ref<DataListButtonElement> create(Document&, DataListButtonOwner&);

private:
    explicit DataListButtonElement(Document&, DataListButtonOwner&);

    void defaultEventHandler(Event&) override;

    DataListButtonOwner& m_owner;
};

} // namespace WebCore
