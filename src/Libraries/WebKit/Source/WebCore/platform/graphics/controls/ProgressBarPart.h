/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 5, 2023.
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

#include "ControlPart.h"
#include <wtf/Seconds.h>

namespace WebCore {

class ProgressBarPart : public ControlPart {
public:
    static Ref<ProgressBarPart> create();
    WEBCORE_EXPORT static Ref<ProgressBarPart> create(double position, const Seconds& animationStartTime);

    double position() const { return m_position; }
    void setPosition(double position) { m_position = position; }

    Seconds animationStartTime() const { return m_animationStartTime; }
    void setAnimationStartTime(Seconds animationStartTime) { m_animationStartTime = animationStartTime; }

private:
    ProgressBarPart(double position, const Seconds& animationStartTime);

    std::unique_ptr<PlatformControl> createPlatformControl() override;

    double m_position;
    Seconds m_animationStartTime;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_CONTROL_PART(ProgressBar)
