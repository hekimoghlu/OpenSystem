/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 18, 2024.
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
// Copyright 2017 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// Resource:
//    Resource lifetime tracking in the Vulkan back-end.
//

#include "libANGLE/renderer/vulkan/vk_resource.h"

#include "libANGLE/renderer/vulkan/ContextVk.h"

namespace rx
{
namespace vk
{
// Resource implementation.
angle::Result Resource::waitForIdle(ContextVk *contextVk,
                                    const char *debugMessage,
                                    RenderPassClosureReason reason)
{
    // If there are pending commands for the resource, flush them.
    if (contextVk->hasUnsubmittedUse(mUse))
    {
        ANGLE_TRY(contextVk->flushAndSubmitCommands(nullptr, nullptr, reason));
    }

    Renderer *renderer = contextVk->getRenderer();
    // Make sure the driver is done with the resource.
    if (!renderer->hasResourceUseFinished(mUse))
    {
        if (debugMessage)
        {
            ANGLE_VK_PERF_WARNING(contextVk, GL_DEBUG_SEVERITY_HIGH, "%s", debugMessage);
        }
        ANGLE_TRY(renderer->finishResourceUse(contextVk, mUse));
    }

    ASSERT(renderer->hasResourceUseFinished(mUse));

    return angle::Result::Continue;
}

std::ostream &operator<<(std::ostream &os, const ResourceUse &use)
{
    const Serials &serials = use.getSerials();
    os << '{';
    for (size_t i = 0; i < serials.size(); i++)
    {
        os << serials[i].getValue();
        if (i < serials.size() - 1)
        {
            os << ",";
        }
    }
    os << '}';
    return os;
}

// SharedGarbage implementation.
SharedGarbage::SharedGarbage() = default;

SharedGarbage::SharedGarbage(SharedGarbage &&other)
{
    *this = std::move(other);
}

SharedGarbage::SharedGarbage(const ResourceUse &use, GarbageObjects &&garbage)
    : mLifetime(use), mGarbage(std::move(garbage))
{}

SharedGarbage::~SharedGarbage() = default;

SharedGarbage &SharedGarbage::operator=(SharedGarbage &&rhs)
{
    std::swap(mLifetime, rhs.mLifetime);
    std::swap(mGarbage, rhs.mGarbage);
    return *this;
}

bool SharedGarbage::destroyIfComplete(Renderer *renderer)
{
    if (renderer->hasResourceUseFinished(mLifetime))
    {
        for (GarbageObject &object : mGarbage)
        {
            object.destroy(renderer);
        }
        return true;
    }
    return false;
}

bool SharedGarbage::hasResourceUseSubmitted(Renderer *renderer) const
{
    return renderer->hasResourceUseSubmitted(mLifetime);
}

// ReleasableResource implementation.
template <class T>
void ReleasableResource<T>::release(Renderer *renderer)
{
    renderer->collectGarbage(mUse, &mObject);
}

template class ReleasableResource<Semaphore>;
}  // namespace vk
}  // namespace rx
