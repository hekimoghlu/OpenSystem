/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 21, 2025.
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
// FramebufferImpl.cpp: Implements the class methods for FramebufferImpl.

#include "libANGLE/renderer/FramebufferImpl.h"

namespace rx
{

namespace
{
angle::Result InitAttachment(const gl::Context *context,
                             const gl::FramebufferAttachment *attachment)
{
    ASSERT(attachment->isAttached());
    if (attachment->initState() == gl::InitState::MayNeedInit)
    {
        ANGLE_TRY(attachment->initializeContents(context));
    }
    return angle::Result::Continue;
}
}  // namespace

angle::Result FramebufferImpl::onLabelUpdate(const gl::Context *context)
{
    return angle::Result::Continue;
}

angle::Result FramebufferImpl::ensureAttachmentsInitialized(
    const gl::Context *context,
    const gl::DrawBufferMask &colorAttachments,
    bool depth,
    bool stencil)
{
    // Default implementation iterates over the attachments and individually initializes them

    for (auto colorIndex : colorAttachments)
    {
        ANGLE_TRY(InitAttachment(context, mState.getColorAttachment(colorIndex)));
    }

    if (depth)
    {
        ANGLE_TRY(InitAttachment(context, mState.getDepthAttachment()));
    }

    if (stencil)
    {
        ANGLE_TRY(InitAttachment(context, mState.getStencilAttachment()));
    }

    return angle::Result::Continue;
}

}  // namespace rx
