/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 8, 2023.
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
#ifndef Object_h
#define Object_h

#include <cstddef>

namespace bmalloc {

class Chunk;
class SmallLine;
class SmallPage;

class Object {
public:
    Object(void*);
    Object(Chunk*, void*);
    Object(Chunk* chunk, size_t offset)
        : m_chunk(chunk)
        , m_offset(offset)
    {
    }
    
    Chunk* chunk() { return m_chunk; }
    size_t offset() { return m_offset; }
    char* address();

    SmallLine* line();
    SmallPage* page();
    
    Object operator+(size_t);
    Object operator-(size_t);
    bool operator<=(const Object&);

private:
    Chunk* m_chunk;
    size_t m_offset;
};

inline Object Object::operator+(size_t offset)
{
    return Object(m_chunk, m_offset + offset);
}

inline Object Object::operator-(size_t offset)
{
    return Object(m_chunk, m_offset - offset);
}

inline bool Object::operator<=(const Object& other)
{
    BASSERT(m_chunk == other.m_chunk);
    return m_offset <= other.m_offset;
}

}; // namespace bmalloc

#endif // Object_h
