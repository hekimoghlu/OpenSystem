/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 19, 2022.
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
// Copyright 2018 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// validationES3.h:
//  Inlined validation functions for OpenGL ES 3.0 entry points.

#ifndef LIBANGLE_VALIDATION_ES3_H_
#define LIBANGLE_VALIDATION_ES3_H_

#include "libANGLE/ErrorStrings.h"
#include "libANGLE/validationES3_autogen.h"

namespace gl
{
bool ValidateTexImageFormatCombination(const Context *context,
                                       angle::EntryPoint entryPoint,
                                       TextureType target,
                                       GLenum internalFormat,
                                       GLenum format,
                                       GLenum type);

bool ValidateES3TexImageParametersBase(const Context *context,
                                       angle::EntryPoint entryPoint,
                                       TextureTarget target,
                                       GLint level,
                                       GLenum internalformat,
                                       bool isCompressed,
                                       bool isSubImage,
                                       GLint xoffset,
                                       GLint yoffset,
                                       GLint zoffset,
                                       GLsizei width,
                                       GLsizei height,
                                       GLsizei depth,
                                       GLint border,
                                       GLenum format,
                                       GLenum type,
                                       GLsizei imageSize,
                                       const void *pixels);

bool ValidateES3TexStorageParametersLevel(const Context *context,
                                          angle::EntryPoint entryPoint,
                                          TextureType target,
                                          GLsizei levels,
                                          GLsizei width,
                                          GLsizei height,
                                          GLsizei depth);

bool ValidateES3TexStorageParametersExtent(const Context *context,
                                           angle::EntryPoint entryPoint,
                                           TextureType target,
                                           GLsizei levels,
                                           GLsizei width,
                                           GLsizei height,
                                           GLsizei depth);

bool ValidateES3TexStorageParametersTexObject(const Context *context,
                                              angle::EntryPoint entryPoint,
                                              TextureType target);

bool ValidateES3TexStorageParametersFormat(const Context *context,
                                           angle::EntryPoint entryPoint,
                                           TextureType target,
                                           GLsizei levels,
                                           GLenum internalformat,
                                           GLsizei width,
                                           GLsizei height,
                                           GLsizei depth);

bool ValidateProgramParameteriBase(const Context *context,
                                   angle::EntryPoint entryPoint,
                                   ShaderProgramID program,
                                   GLenum pname,
                                   GLint value);
}  // namespace gl

#endif  // LIBANGLE_VALIDATION_ES3_H_
