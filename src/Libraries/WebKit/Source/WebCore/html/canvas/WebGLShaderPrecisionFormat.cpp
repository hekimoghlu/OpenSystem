/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 2, 2022.
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

#if ENABLE(WEBGL)

#include "WebGLShaderPrecisionFormat.h"
#include <wtf/Ref.h>

namespace WebCore {

// static
Ref<WebGLShaderPrecisionFormat> WebGLShaderPrecisionFormat::create(GCGLint rangeMin, GCGLint rangeMax, GCGLint precision)
{
    return adoptRef(*new WebGLShaderPrecisionFormat(rangeMin, rangeMax, precision));
}

GCGLint WebGLShaderPrecisionFormat::rangeMin() const
{
    return m_rangeMin;
}

GCGLint WebGLShaderPrecisionFormat::rangeMax() const
{
    return m_rangeMax;
}

GCGLint WebGLShaderPrecisionFormat::precision() const
{
    return m_precision;
}

WebGLShaderPrecisionFormat::WebGLShaderPrecisionFormat(GCGLint rangeMin, GCGLint rangeMax, GCGLint precision)
    : m_rangeMin(rangeMin)
    , m_rangeMax(rangeMax)
    , m_precision(precision)
{
}

}

#endif // ENABLE(WEBGL)

