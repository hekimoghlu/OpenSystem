/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 20, 2023.
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
#include "VTTRegionList.h"

#if ENABLE(VIDEO)

namespace WebCore {

VTTRegion* VTTRegionList::item(unsigned index) const
{
    if (index >= m_vector.size())
        return nullptr;
    return const_cast<VTTRegion*>(m_vector[index].ptr());
}

VTTRegion* VTTRegionList::getRegionById(const String& id) const
{
    // FIXME: Why is this special case needed?
    if (id.isEmpty())
        return nullptr;
    for (auto& region : m_vector) {
        if (region->id() == id)
            return const_cast<VTTRegion*>(region.ptr());
    }
    return nullptr;
}

void VTTRegionList::add(Ref<VTTRegion>&& region)
{
    m_vector.append(WTFMove(region));
}

void VTTRegionList::remove(VTTRegion& region)
{
    for (unsigned i = 0, size = m_vector.size(); i < size; ++i) {
        if (m_vector[i].ptr() == &region) {
            m_vector.remove(i);
            return;
        }
    }
    ASSERT_NOT_REACHED();
}

} // namespace WebCore

#endif
