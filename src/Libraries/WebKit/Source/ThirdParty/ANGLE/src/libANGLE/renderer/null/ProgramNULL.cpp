/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 15, 2023.
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
// Copyright 2016 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// ProgramNULL.cpp:
//    Implements the class methods for ProgramNULL.
//

#include "libANGLE/renderer/null/ProgramNULL.h"

#include "common/debug.h"

namespace rx
{
namespace
{
class LinkTaskNULL : public LinkTask
{
  public:
    LinkTaskNULL(const gl::ProgramState *state) : mState(state) {}
    ~LinkTaskNULL() override = default;
    void link(const gl::ProgramLinkedResources &resources,
              const gl::ProgramMergedVaryings &mergedVaryings,
              std::vector<std::shared_ptr<LinkSubTask>> *linkSubTasksOut,
              std::vector<std::shared_ptr<LinkSubTask>> *postLinkSubTasksOut) override
    {
        ASSERT(linkSubTasksOut && linkSubTasksOut->empty());
        ASSERT(postLinkSubTasksOut && postLinkSubTasksOut->empty());

        const gl::SharedCompiledShaderState &fragmentShader =
            mState->getAttachedShader(gl::ShaderType::Fragment);
        if (fragmentShader != nullptr)
        {
            resources.pixelLocalStorageLinker.link(fragmentShader->pixelLocalStorageFormats);
        }

        return;
    }
    angle::Result getResult(const gl::Context *context, gl::InfoLog &infoLog) override
    {
        return angle::Result::Continue;
    }

  private:
    const gl::ProgramState *mState;
};
}  // anonymous namespace

ProgramNULL::ProgramNULL(const gl::ProgramState &state) : ProgramImpl(state) {}

ProgramNULL::~ProgramNULL() {}

angle::Result ProgramNULL::load(const gl::Context *context,
                                gl::BinaryInputStream *stream,
                                std::shared_ptr<LinkTask> *loadTaskOut,
                                egl::CacheGetResult *resultOut)
{
    *loadTaskOut = {};
    *resultOut   = egl::CacheGetResult::Success;
    return angle::Result::Continue;
}

void ProgramNULL::save(const gl::Context *context, gl::BinaryOutputStream *stream) {}

void ProgramNULL::setBinaryRetrievableHint(bool retrievable) {}

void ProgramNULL::setSeparable(bool separable) {}

angle::Result ProgramNULL::link(const gl::Context *contextImpl,
                                std::shared_ptr<LinkTask> *linkTaskOut)
{
    *linkTaskOut = std::shared_ptr<LinkTask>(new LinkTaskNULL(&mState));
    return angle::Result::Continue;
}

GLboolean ProgramNULL::validate(const gl::Caps &caps)
{
    return GL_TRUE;
}

}  // namespace rx
