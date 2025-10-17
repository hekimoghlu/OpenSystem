/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 21, 2022.
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
#include "WebExtensionDataRecord.h"

#if ENABLE(WK_WEB_EXTENSIONS)

#include <wtf/OptionSet.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(WebExtensionDataRecordHolder);

WebExtensionDataRecord::WebExtensionDataRecord(const String& displayName, const String& uniqueIdentifier)
    : m_displayName(displayName)
    , m_uniqueIdentifier(uniqueIdentifier)
{
}

bool WebExtensionDataRecord::operator==(const WebExtensionDataRecord& other) const
{
    return this == &other || (m_displayName == other.m_displayName && m_uniqueIdentifier == other.m_uniqueIdentifier);
}

size_t WebExtensionDataRecord::totalSize() const
{
    size_t total = 0;
    for (auto& entry : m_typeSizes)
        total += entry.value;
    return total;
}

size_t WebExtensionDataRecord::sizeOfTypes(OptionSet<Type> types) const
{
    size_t total = 0;
    for (auto type : types)
        total += m_typeSizes.get(type);
    return total;
}

OptionSet<WebExtensionDataRecord::Type> WebExtensionDataRecord::types() const
{
    OptionSet<WebExtensionDataRecord::Type> result;
    for (auto& entry : m_typeSizes)
        result.add(entry.key);
    return result;
}

} // namespace WebKit

#endif // ENABLE(WK_WEB_EXTENSIONS)
