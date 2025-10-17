/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 16, 2024.
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
// Copyright 2024 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// ProgramWgpu.h:
//    Defines the class interface for ProgramWgpu, implementing ProgramImpl.
//

#ifndef LIBANGLE_RENDERER_WGPU_PROGRAMWGPU_H_
#define LIBANGLE_RENDERER_WGPU_PROGRAMWGPU_H_

#include "libANGLE/renderer/ProgramImpl.h"

namespace rx
{

class ProgramWgpu : public ProgramImpl
{
  public:
    ProgramWgpu(const gl::ProgramState &state);
    ~ProgramWgpu() override;

    angle::Result load(const gl::Context *context,
                       gl::BinaryInputStream *stream,
                       std::shared_ptr<LinkTask> *loadTaskOut,
                       egl::CacheGetResult *resultOut) override;
    void save(const gl::Context *context, gl::BinaryOutputStream *stream) override;
    void setBinaryRetrievableHint(bool retrievable) override;
    void setSeparable(bool separable) override;

    angle::Result link(const gl::Context *context, std::shared_ptr<LinkTask> *linkTaskOut) override;
    GLboolean validate(const gl::Caps &caps) override;
};

}  // namespace rx

#endif  // LIBANGLE_RENDERER_WGPU_PROGRAMWGPU_H_
