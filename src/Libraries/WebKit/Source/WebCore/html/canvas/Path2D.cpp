/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 11, 2024.
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
#include "Path2D.h"

#include "AffineTransform.h"
#include "DOMMatrix2DInit.h"
#include "DOMMatrixReadOnly.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(Path2D);

Path2D::~Path2D() = default;

ExceptionOr<void> Path2D::addPath(Path2D& path, DOMMatrix2DInit&& matrixInit)
{
    auto checkValid = DOMMatrixReadOnly::validateAndFixup(matrixInit);
    if (checkValid.hasException())
        return checkValid.releaseException();

    m_path.addPath(path.path(), { matrixInit.a.value_or(1), matrixInit.b.value_or(0), matrixInit.c.value_or(0), matrixInit.d.value_or(1), matrixInit.e.value_or(0), matrixInit.f.value_or(0) });
    return { };
}

}
