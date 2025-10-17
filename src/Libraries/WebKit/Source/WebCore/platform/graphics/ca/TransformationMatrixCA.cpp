/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 1, 2025.
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

#if USE(CA)

#include "FloatConversion.h"
#include <QuartzCore/CATransform3D.h>

namespace WebCore {

TransformationMatrix::TransformationMatrix(const CATransform3D& t)
{
    setMatrix(
        t.m11, t.m12, t.m13, t.m14,
        t.m21, t.m22, t.m23, t.m24, 
        t.m31, t.m32, t.m33, t.m34, 
        t.m41, t.m42, t.m43, t.m44);
}

TransformationMatrix::operator CATransform3D() const
{
    CATransform3D toT3D;
    toT3D.m11 = narrowPrecisionToFloat(m11());
    toT3D.m12 = narrowPrecisionToFloat(m12());
    toT3D.m13 = narrowPrecisionToFloat(m13());
    toT3D.m14 = narrowPrecisionToFloat(m14());
    toT3D.m21 = narrowPrecisionToFloat(m21());
    toT3D.m22 = narrowPrecisionToFloat(m22());
    toT3D.m23 = narrowPrecisionToFloat(m23());
    toT3D.m24 = narrowPrecisionToFloat(m24());
    toT3D.m31 = narrowPrecisionToFloat(m31());
    toT3D.m32 = narrowPrecisionToFloat(m32());
    toT3D.m33 = narrowPrecisionToFloat(m33());
    toT3D.m34 = narrowPrecisionToFloat(m34());
    toT3D.m41 = narrowPrecisionToFloat(m41());
    toT3D.m42 = narrowPrecisionToFloat(m42());
    toT3D.m43 = narrowPrecisionToFloat(m43());
    toT3D.m44 = narrowPrecisionToFloat(m44());
    return toT3D;
}

}

#endif // USE(CA)
