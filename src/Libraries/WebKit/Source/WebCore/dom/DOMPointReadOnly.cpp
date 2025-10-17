/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 30, 2022.
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
#include "DOMPointReadOnly.h"

#include "DOMMatrixReadOnly.h"
#include "DOMPoint.h"
#include "WebCoreOpaqueRoot.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(DOMPointReadOnly);
    
ExceptionOr<Ref<DOMPoint>> DOMPointReadOnly::matrixTransform(DOMMatrixInit&& matrixInit) const
{
    auto matrixOrException = DOMMatrixReadOnly::fromMatrix(WTFMove(matrixInit));
    if (matrixOrException.hasException())
        return matrixOrException.releaseException();

    auto matrix = matrixOrException.releaseReturnValue();
    
    double x = this->x();
    double y = this->y();
    double z = this->z();
    double w = this->w();
    matrix->transformationMatrix().map4ComponentPoint(x, y, z, w);
    
    return { DOMPoint::create(x, y, z, w) };
}

WebCoreOpaqueRoot root(DOMPointReadOnly* point)
{
    return WebCoreOpaqueRoot { point };
}

} // namespace WebCore

