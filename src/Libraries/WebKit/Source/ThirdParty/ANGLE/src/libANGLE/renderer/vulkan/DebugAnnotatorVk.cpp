/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 4, 2023.
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
// DebugAnnotatorVk.cpp: Vulkan helpers for adding trace annotations.
//

#include "libANGLE/renderer/vulkan/DebugAnnotatorVk.h"

#include "common/entry_points_enum_autogen.h"
#include "libANGLE/Context.h"
#include "libANGLE/renderer/vulkan/ContextVk.h"

namespace rx
{

DebugAnnotatorVk::DebugAnnotatorVk() {}

DebugAnnotatorVk::~DebugAnnotatorVk() {}

void DebugAnnotatorVk::beginEvent(gl::Context *context,
                                  angle::EntryPoint entryPoint,
                                  const char *eventName,
                                  const char *eventMessage)
{
    angle::LoggingAnnotator::beginEvent(context, entryPoint, eventName, eventMessage);
    if (vkCmdBeginDebugUtilsLabelEXT && context)
    {
        ContextVk *contextVk = vk::GetImpl(static_cast<gl::Context *>(context));
        contextVk->logEvent(eventMessage);
    }
}

void DebugAnnotatorVk::endEvent(gl::Context *context,
                                const char *eventName,
                                angle::EntryPoint entryPoint)
{
    angle::LoggingAnnotator::endEvent(context, eventName, entryPoint);
    if (vkCmdBeginDebugUtilsLabelEXT && context)
    {
        ContextVk *contextVk = vk::GetImpl(static_cast<gl::Context *>(context));
        if (angle::IsDrawEntryPoint(entryPoint))
        {
            contextVk->endEventLog(entryPoint, PipelineType::Graphics);
        }
        else if (angle::IsDispatchEntryPoint(entryPoint))
        {
            contextVk->endEventLog(entryPoint, PipelineType::Compute);
        }
        else if (angle::IsClearEntryPoint(entryPoint) || angle::IsQueryEntryPoint(entryPoint))
        {
            contextVk->endEventLogForClearOrQuery();
        }
    }
}

bool DebugAnnotatorVk::getStatus(const gl::Context *context)
{
    return true;
}
}  // namespace rx
