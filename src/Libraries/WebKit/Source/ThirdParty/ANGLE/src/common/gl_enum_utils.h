/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 1, 2024.
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
// Copyright 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// gl_enum_utils.h:
//   Utility functions for converting GLenums to string.

#ifndef LIBANGLE_GL_ENUM_UTILS_H_
#define LIBANGLE_GL_ENUM_UTILS_H_

#include <ostream>
#include <string>

#include "common/gl_enum_utils_autogen.h"

namespace gl
{
const char *GLbooleanToString(unsigned int value);
const char *GLenumToString(GLESEnum enumGroup, unsigned int value);
const char *GLenumToString(BigGLEnum enumGroup, unsigned int value);
std::string GLbitfieldToString(GLESEnum enumGroup, unsigned int value);
std::string GLbitfieldToString(BigGLEnum enumGroup, unsigned int value);
void OutputGLenumString(std::ostream &out, GLESEnum enumGroup, unsigned int value);
void OutputGLenumString(std::ostream &out, BigGLEnum enumGroup, unsigned int value);
void OutputGLbitfieldString(std::ostream &out, GLESEnum enumGroup, unsigned int value);
const char *GLinternalFormatToString(unsigned int format);
unsigned int StringToGLenum(const char *str);
unsigned int StringToGLbitfield(const char *str);

extern const char kUnknownGLenumString[];
}  // namespace gl

#endif  // LIBANGLE_GL_ENUM_UTILS_H_
