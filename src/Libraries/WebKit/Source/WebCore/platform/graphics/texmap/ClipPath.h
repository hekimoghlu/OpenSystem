/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 15, 2024.
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

#include "FloatPoint.h"
#include "FloatRect.h"
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>

namespace WebCore {

class ClipPath final {
    WTF_MAKE_TZONE_ALLOCATED(ClipPath);
public:
    ClipPath() = default;
    ClipPath(Vector<FloatPoint>&& vertices, unsigned bufferID, unsigned bufferOffsetInBytes);

    bool isEmpty() const { return m_vertices.isEmpty(); }
    unsigned bufferID() const { return m_bufferID; }
    const void* bufferDataOffsetAsPtr() const;
    unsigned numberOfVertices() const { return m_vertices.size(); }

    const FloatRect& bounds() const { return m_bounds; }

private:

    const Vector<FloatPoint> m_vertices;
    unsigned m_bufferID { 0 };
    unsigned m_bufferOffsetInBytes { 0 };

    FloatRect m_bounds;
};

} // namespace WebCore
