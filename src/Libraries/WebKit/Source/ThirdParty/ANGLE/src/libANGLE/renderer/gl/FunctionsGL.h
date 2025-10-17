/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 11, 2023.
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

// FunctionsGL.h: Defines the FuntionsGL class to contain loaded GL functions

#ifndef LIBANGLE_RENDERER_GL_FUNCTIONSGL_H_
#define LIBANGLE_RENDERER_GL_FUNCTIONSGL_H_

#include "common/debug.h"
#include "libANGLE/Version.h"
#include "libANGLE/renderer/gl/DispatchTableGL_autogen.h"
#include "libANGLE/renderer/gl/functionsgl_enums.h"
#include "libANGLE/renderer/gl/functionsgl_typedefs.h"

namespace egl
{
class AttributeMap;
}  // namespace egl

namespace rx
{

enum StandardGL
{
    STANDARD_GL_DESKTOP,
    STANDARD_GL_ES,
};

class FunctionsGL : public DispatchTableGL
{
  public:
    FunctionsGL();
    ~FunctionsGL() override;

    void initialize(const egl::AttributeMap &displayAttributes);

    // Version information
    gl::Version version;
    StandardGL standard;
    GLint profile;
    bool isAtLeastGL(const gl::Version &glVersion) const;
    bool isAtMostGL(const gl::Version &glVersion) const;
    bool isAtLeastGLES(const gl::Version &glesVersion) const;
    bool isAtMostGLES(const gl::Version &glesVersion) const;

    // Extensions
    std::vector<std::string> extensions;
    bool hasExtension(const std::string &ext) const;
    bool hasGLExtension(const std::string &ext) const;
    bool hasGLESExtension(const std::string &ext) const;

  private:
    void *loadProcAddress(const std::string &function) const override = 0;
    void initializeStubFunctionsForNULLDriver(const std::set<std::string> &extensionSet);
};

}  // namespace rx

#endif  // LIBANGLE_RENDERER_GL_FUNCTIONSGL_H_
