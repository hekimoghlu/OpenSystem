/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 3, 2022.
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
// Copyright 2014 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// CompilerImpl.h: Defines the rx::CompilerImpl class, an implementation interface
//                 for the gl::Compiler object.

#include "GLSLANG/ShaderLang.h"
#include "common/angleutils.h"
#include "libANGLE/Error.h"

#ifndef LIBANGLE_RENDERER_COMPILERIMPL_H_
#    define LIBANGLE_RENDERER_COMPILERIMPL_H_

namespace rx
{

class CompilerImpl : angle::NonCopyable
{
  public:
    CompilerImpl() {}
    virtual ~CompilerImpl() {}

    // TODO(jmadill): Expose translator built-in resources init method.
    virtual ShShaderOutput getTranslatorOutputType() const = 0;
};

}  // namespace rx

#endif  // LIBANGLE_RENDERER_COMPILERIMPL_H_
