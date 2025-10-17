/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 29, 2022.
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

//
// Copyright 2015 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// VertexAttribute.inc: Inline vertex attribute methods
//

namespace gl
{

inline VertexAttribCurrentValueData::VertexAttribCurrentValueData()
    : Type(gl::VertexAttribType::Float)
{
    Values.FloatValues[0] = 0.0f;
    Values.FloatValues[1] = 0.0f;
    Values.FloatValues[2] = 0.0f;
    Values.FloatValues[3] = 1.0f;
}

inline void VertexAttribCurrentValueData::setFloatValues(const GLfloat floatValues[4])
{
    for (unsigned int valueIndex = 0; valueIndex < 4; valueIndex++)
    {
        Values.FloatValues[valueIndex] = floatValues[valueIndex];
    }
    Type = gl::VertexAttribType::Float;
}

inline void VertexAttribCurrentValueData::setIntValues(const GLint intValues[4])
{
    for (unsigned int valueIndex = 0; valueIndex < 4; valueIndex++)
    {
        Values.IntValues[valueIndex] = intValues[valueIndex];
    }
    Type = gl::VertexAttribType::Int;
}

inline void VertexAttribCurrentValueData::setUnsignedIntValues(const GLuint unsignedIntValues[4])
{
    for (unsigned int valueIndex = 0; valueIndex < 4; valueIndex++)
    {
        Values.UnsignedIntValues[valueIndex] = unsignedIntValues[valueIndex];
    }
    Type = gl::VertexAttribType::UnsignedInt;
}

inline bool operator==(const VertexAttribCurrentValueData &a, const VertexAttribCurrentValueData &b)
{
    return (a.Type == b.Type && memcmp(&a.Values, &b.Values, sizeof(VertexAttribCurrentValueData::Values)) == 0);
}

inline bool operator!=(const VertexAttribCurrentValueData &a, const VertexAttribCurrentValueData &b)
{
    return !(a == b);
}

}  // namespace gl
