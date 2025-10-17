/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 27, 2023.
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
// Program11: D3D11 implementation of an OpenGL Program.

#ifndef LIBANGLE_RENDERER_D3D_D3D11_PROGRAM11_H_
#define LIBANGLE_RENDERER_D3D_D3D11_PROGRAM11_H_

#include "libANGLE/renderer/d3d/ProgramD3D.h"

namespace rx
{
class Renderer11;

class Program11 : public ProgramD3D
{
  public:
    Program11(const gl::ProgramState &programState, Renderer11 *renderer11);
    ~Program11() override;
};
}  // namespace rx

#endif  // LIBANGLE_RENDERER_D3D_D3D11_PROGRAM11_H_
