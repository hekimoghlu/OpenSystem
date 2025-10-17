/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 19, 2022.
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

#include "WebGLUniformLocation.h"

namespace WebCore {

Ref<WebGLUniformLocation> WebGLUniformLocation::create(WebGLProgram* program, GCGLint location, GCGLenum type)
{
    return adoptRef(*new WebGLUniformLocation(program, location, type));
}

WebGLUniformLocation::WebGLUniformLocation(WebGLProgram* program, GCGLint location, GCGLenum type)
    : m_program(program)
    , m_location(location)
    , m_type(type)
{
    ASSERT(m_program);
    m_linkCount = m_program->getLinkCount();
}

WebGLProgram* WebGLUniformLocation::program() const
{
    // If the program has been linked again, then this UniformLocation is no
    // longer valid.
    if (m_program->getLinkCount() != m_linkCount)
        return 0;
    return m_program.get();
}

GCGLint WebGLUniformLocation::location() const
{
    // If the program has been linked again, then this UniformLocation is no
    // longer valid.
    ASSERT(m_program->getLinkCount() == m_linkCount);
    return m_location;
}
    
GCGLenum WebGLUniformLocation::type() const
{
    // If the program has been linked again, then this UniformLocation is no
    // longer valid.
    ASSERT(m_program->getLinkCount() == m_linkCount);
    return m_type;
}

}

#endif // ENABLE(WEBGL)
