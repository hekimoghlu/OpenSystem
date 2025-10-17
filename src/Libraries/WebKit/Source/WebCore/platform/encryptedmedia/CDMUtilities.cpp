/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 30, 2023.
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
#include "CDMUtilities.h"

#if ENABLE(ENCRYPTED_MEDIA)

#include "SharedBuffer.h"
#include <wtf/text/WTFString.h>

namespace WebCore {

namespace CDMUtilities {

RefPtr<JSON::Object> parseJSONObject(const SharedBuffer& buffer)
{
    // Fail on large buffers whose size doesn't fit into a 32-bit unsigned integer.
    if (buffer.size() > std::numeric_limits<unsigned>::max())
        return nullptr;

    // Parse the buffer contents as JSON, returning the root object (if any).
    String json { buffer.span() };

    auto value = JSON::Value::parseJSON(json);
    if (!value)
        return nullptr;

    return value->asObject();
}

};

};

#endif // ENABLE(ENCRYPTED_MEDIA)
