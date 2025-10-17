/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 7, 2025.
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
// Copyright 2022 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// ProgramImpl.cpp: Implements the class methods for ProgramImpl.

#include "libANGLE/renderer/ProgramImpl.h"

namespace rx
{
void LinkTask::link(const gl::ProgramLinkedResources &resources,
                    const gl::ProgramMergedVaryings &mergedVaryings,
                    std::vector<std::shared_ptr<LinkSubTask>> *linkSubTasksOut,
                    std::vector<std::shared_ptr<LinkSubTask>> *postLinkSubTasksOut)
{
    UNREACHABLE();
    return;
}
void LinkTask::load(std::vector<std::shared_ptr<LinkSubTask>> *linkSubTasksOut,
                    std::vector<std::shared_ptr<LinkSubTask>> *postLinkSubTasksOut)
{
    UNREACHABLE();
    return;
}
bool LinkTask::isLinkingInternally()
{
    return false;
}

angle::Result ProgramImpl::onLabelUpdate(const gl::Context *context)
{
    return angle::Result::Continue;
}

}  // namespace rx
