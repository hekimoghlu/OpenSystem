/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 3, 2022.
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
#include "TextureMapperGPUBuffer.h"

namespace WebCore {

TextureMapperGPUBuffer::TextureMapperGPUBuffer(size_t size, Type type, Usage usage)
    : m_size(size)
    , m_target((type == Type::Vertex) ? GL_ARRAY_BUFFER : GL_ELEMENT_ARRAY_BUFFER)
    , m_usage((usage == Usage::Dynamic) ? GL_DYNAMIC_DRAW : GL_STATIC_DRAW)
{
    if (!size)
        return;

    glGenBuffers(1, &m_id);
    if (m_id) {
        glBindBuffer(m_target, m_id);
        glBufferData(m_target, m_size, nullptr, m_usage);
        if (glGetError() != GL_NO_ERROR) {
            glDeleteBuffers(1, &m_id);
            m_id = 0;
        }
    }
}

TextureMapperGPUBuffer::~TextureMapperGPUBuffer()
{
    if (m_id) {
        glDeleteBuffers(1, &m_id);
        m_id = 0;
    }
}

bool TextureMapperGPUBuffer::updateData(const void* data, size_t offset, size_t size)
{
    if (m_id) {
        glBindBuffer(m_target, m_id);
        glBufferData(m_target, m_size, nullptr, m_usage); // Invalidate. No need to preserve previous content
        if (glGetError() != GL_NO_ERROR)
            return false;
        glBufferSubData(m_target, offset, size, data);
        return true;
    }
    return false;
}

} // namespace WebCore
