/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 11, 2023.
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

#include <wtf/Forward.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

class Document;
class Element;

class ValidationMessageClient {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(ValidationMessageClient);
public:
    virtual ~ValidationMessageClient() = default;

    // Show validation message for the specified anchor element. An
    // implementation of this function may hide the message automatically after
    // some period.
    virtual void showValidationMessage(const Element& anchor, const String& message) = 0;

    // Hide validation message for the specified anchor if the message for the
    // anchor is already visible.
    virtual void hideValidationMessage(const Element& anchor) = 0;

    // Hide any validation message currently displayed.
    virtual void hideAnyValidationMessage() = 0;

    // Returns true if the validation message for the specified anchor element
    // is visible.
    virtual bool isValidationMessageVisible(const Element& anchor) = 0;

    virtual void updateValidationBubbleStateIfNeeded() = 0;

    virtual void documentDetached(Document&) = 0;
};

} // namespace WebCore
