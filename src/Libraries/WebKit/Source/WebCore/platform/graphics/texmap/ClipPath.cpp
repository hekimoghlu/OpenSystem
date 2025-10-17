/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 23, 2025.
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
#include "ClipPath.h"

#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(ClipPath);

ClipPath::ClipPath(Vector<FloatPoint>&& vertices, unsigned bufferID, unsigned bufferOffsetInBytes)
    : m_vertices(WTFMove(vertices))
    , m_bufferID(bufferID)
    , m_bufferOffsetInBytes(bufferOffsetInBytes)
{
    unsigned vertexCount = numberOfVertices();
    if (vertexCount) {
        m_bounds.setLocation(m_vertices.at(0));
        for (size_t i = 1; i < vertexCount; i++)
            m_bounds.extend(m_vertices.at(i));
    }
}

const void* ClipPath::bufferDataOffsetAsPtr() const
{
    if (m_bufferID)
        return reinterpret_cast<const void*>(m_bufferOffsetInBytes);

    return nullptr;
}

} // namespace WebCore
