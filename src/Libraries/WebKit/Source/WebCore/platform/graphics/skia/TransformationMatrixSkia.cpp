/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 26, 2025.
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
#include "TransformationMatrix.h"

#if USE(SKIA)
#include "AffineTransform.h"
WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_BEGIN // GLib/Win ports
#include <skia/core/SkM44.h>
WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_END

namespace WebCore {

TransformationMatrix::TransformationMatrix(const SkM44& t)
    : TransformationMatrix(SkScalarToDouble(t.rc(0, 0)), SkScalarToDouble(t.rc(0, 1)), SkScalarToDouble(t.rc(0, 2)), SkScalarToDouble(t.rc(0, 3)),
        SkScalarToDouble(t.rc(1, 0)), SkScalarToDouble(t.rc(1, 1)), SkScalarToDouble(t.rc(1, 2)), SkScalarToDouble(t.rc(1, 3)),
        SkScalarToDouble(t.rc(2, 0)), SkScalarToDouble(t.rc(2, 1)), SkScalarToDouble(t.rc(2, 2)), SkScalarToDouble(t.rc(2, 3)),
        SkScalarToDouble(t.rc(3, 0)), SkScalarToDouble(t.rc(3, 1)), SkScalarToDouble(t.rc(3, 2)), SkScalarToDouble(t.rc(3, 3)))
{
}

TransformationMatrix::operator SkM44() const
{
    return SkM44 {
        SkDoubleToScalar(m11()),
        SkDoubleToScalar(m12()),
        SkDoubleToScalar(m13()),
        SkDoubleToScalar(m14()),
        SkDoubleToScalar(m21()),
        SkDoubleToScalar(m22()),
        SkDoubleToScalar(m23()),
        SkDoubleToScalar(m24()),
        SkDoubleToScalar(m31()),
        SkDoubleToScalar(m32()),
        SkDoubleToScalar(m33()),
        SkDoubleToScalar(m34()),
        SkDoubleToScalar(m41()),
        SkDoubleToScalar(m42()),
        SkDoubleToScalar(m43()),
        SkDoubleToScalar(m44())
    };
}

AffineTransform::AffineTransform(const SkMatrix& t)
    : AffineTransform(SkScalarToDouble(t[SkMatrix::kMScaleX]), SkScalarToDouble(t[SkMatrix::kMSkewY]), SkScalarToDouble(t[SkMatrix::kMSkewX]),  SkScalarToDouble(t[SkMatrix::kMScaleY]),
        SkScalarToDouble(t[SkMatrix::kMTransX]), SkScalarToDouble(t[SkMatrix::kMTransY]))
{
}

AffineTransform::operator SkMatrix() const
{
    return SkMatrix::MakeAll(SkDoubleToScalar(a()), SkDoubleToScalar(c()), SkDoubleToScalar(e()), SkDoubleToScalar(b()), SkDoubleToScalar(d()), SkDoubleToScalar(f()), 0, 0, SK_Scalar1);
}

} // namespace WebCore

#endif // USE(SKIA)
