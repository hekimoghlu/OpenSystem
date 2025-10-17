/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 23, 2025.
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
// Copyright 2020 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// glx_utils.h: Utility routines specific to the G:X->EGL implementation.

#ifndef LIBANGLE_RENDERER_GL_GLX_GLXUTILS_H_
#define LIBANGLE_RENDERER_GL_GLX_GLXUTILS_H_

#include <string>

#include "common/platform.h"
#include "libANGLE/renderer/gl/glx/FunctionsGLX.h"

namespace rx
{

namespace x11
{

std::string XErrorToString(Display *display, int status);

}  // namespace x11

}  // namespace rx

#endif  // LIBANGLE_RENDERER_GL_GLX_GLXUTILS_H_
