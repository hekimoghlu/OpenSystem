/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 15, 2024.
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
#pragma once

#include <wtf/Assertions.h>

namespace WebCore {

enum class ImmediateActionStage : uint8_t {
    None,
    PerformedHitTest,
    ActionUpdated,
    ActionCancelledWithoutUpdate,
    ActionCancelledAfterUpdate,
    ActionCompleted,
};

constexpr bool immediateActionBeganOrWasCompleted(ImmediateActionStage immediateActionStage)
{
    switch (immediateActionStage) {
    case ImmediateActionStage::ActionCompleted:
    case ImmediateActionStage::ActionUpdated:
    case ImmediateActionStage::ActionCancelledAfterUpdate:
        return true;
    case ImmediateActionStage::None:
    case ImmediateActionStage::PerformedHitTest:
    case ImmediateActionStage::ActionCancelledWithoutUpdate:
        return false;
    }
    ASSERT_NOT_REACHED();
    return false;
}

} // namespace WebCore
