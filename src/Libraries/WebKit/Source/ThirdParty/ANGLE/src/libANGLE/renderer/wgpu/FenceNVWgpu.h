/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 2, 2023.
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
// FenceNVWgpu.h:
//    Defines the class interface for FenceNVWgpu, implementing FenceNVImpl.
//

#ifndef LIBANGLE_RENDERER_WGPU_FENCENVWGPU_H_
#define LIBANGLE_RENDERER_WGPU_FENCENVWGPU_H_

#include "libANGLE/renderer/FenceNVImpl.h"

namespace rx
{
class FenceNVWgpu : public FenceNVImpl
{
  public:
    FenceNVWgpu();
    ~FenceNVWgpu() override;

    void onDestroy(const gl::Context *context) override {}
    angle::Result set(const gl::Context *context, GLenum condition) override;
    angle::Result test(const gl::Context *context, GLboolean *outFinished) override;
    angle::Result finish(const gl::Context *context) override;
};
}  // namespace rx

#endif  // LIBANGLE_RENDERER_WGPU_FENCENVWGPU_H_
