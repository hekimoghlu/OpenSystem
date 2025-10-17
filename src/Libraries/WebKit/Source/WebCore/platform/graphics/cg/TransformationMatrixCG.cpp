/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 31, 2023.
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
#include "AffineTransform.h"
#include "TransformationMatrix.h"

#if USE(CG)

#include <CoreGraphics/CGAffineTransform.h>
#include "FloatConversion.h"

namespace WebCore {

TransformationMatrix::TransformationMatrix(const CGAffineTransform& t)
{
    setA(t.a);
    setB(t.b);
    setC(t.c);
    setD(t.d);
    setE(t.tx);
    setF(t.ty);
}

TransformationMatrix::operator CGAffineTransform() const
{
    return CGAffineTransformMake(narrowPrecisionToCGFloat(a()),
                                 narrowPrecisionToCGFloat(b()),
                                 narrowPrecisionToCGFloat(c()),
                                 narrowPrecisionToCGFloat(d()),
                                 narrowPrecisionToCGFloat(e()),
                                 narrowPrecisionToCGFloat(f()));
}

AffineTransform::AffineTransform(const CGAffineTransform& t)
{
    setMatrix(t.a, t.b, t.c, t.d, t.tx, t.ty);
}

AffineTransform::operator CGAffineTransform() const
{
    return CGAffineTransformMake(narrowPrecisionToCGFloat(a()),
                                 narrowPrecisionToCGFloat(b()),
                                 narrowPrecisionToCGFloat(c()),
                                 narrowPrecisionToCGFloat(d()),
                                 narrowPrecisionToCGFloat(e()),
                                 narrowPrecisionToCGFloat(f()));
}

}

#endif // USE(CG)
