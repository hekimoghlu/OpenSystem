/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 10, 2025.
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
#include <wtf/RefPtr.h>

namespace WebCore {

class SharedBuffer;

struct PromisedAttachmentInfo {
#if ENABLE(ATTACHMENT_ELEMENT)
    String attachmentIdentifier;
#endif

    Vector<std::pair<String, RefPtr<WebCore::SharedBuffer>>> additionalTypesAndData;

    operator bool() const
    {
#if ENABLE(ATTACHMENT_ELEMENT)
        if (!attachmentIdentifier.isEmpty())
            return true;
#endif
        return false;
    }
};

} // namespace WebCore

