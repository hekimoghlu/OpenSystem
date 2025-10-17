/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 7, 2022.
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

#include "CanvasPath.h"
#include "SVGPathUtilities.h"
#include <wtf/RefCounted.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

struct DOMMatrix2DInit;

class WEBCORE_EXPORT Path2D final : public RefCounted<Path2D>, public CanvasPath {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(Path2D, WEBCORE_EXPORT);
public:
    virtual ~Path2D();

    static Ref<Path2D> create()
    {
        return adoptRef(*new Path2D);
    }

    static Ref<Path2D> create(const Path& path)
    {
        return adoptRef(*new Path2D(path));
    }

    static Ref<Path2D> create(const Path2D& path)
    {
        return create(path.path());
    }

    static Ref<Path2D> create(StringView pathData)
    {
        return create(buildPathFromString(pathData));
    }

    ExceptionOr<void> addPath(Path2D&, DOMMatrix2DInit&&);

    const Path& path() const { return m_path; }

private:
    Path2D() = default;
    Path2D(const Path& path)
        : CanvasPath(path)
    {
    }
};

} // namespace WebCore
