/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 3, 2023.
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
#include "config.h"
#include "ProgressBarPart.h"

#include "ControlFactory.h"

namespace WebCore {

Ref<ProgressBarPart> ProgressBarPart::create()
{
    return adoptRef(*new ProgressBarPart(0, 0_s));
}

Ref<ProgressBarPart> ProgressBarPart::create(double position, const Seconds& animationStartTime)
{
    return adoptRef(*new ProgressBarPart(position, animationStartTime));
}

ProgressBarPart::ProgressBarPart(double position, const Seconds& animationStartTime)
    : ControlPart(StyleAppearance::ProgressBar)
    , m_position(position)
    , m_animationStartTime(animationStartTime)
{
}

std::unique_ptr<PlatformControl> ProgressBarPart::createPlatformControl()
{
    return controlFactory().createPlatformProgressBar(*this);
}

} // namespace WebCore
