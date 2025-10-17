/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 29, 2023.
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

#include "ControlFactory.h"
#include "ControlPart.h"

namespace WebCore {

class SwitchThumbPart final : public ControlPart {
public:
    static Ref<SwitchThumbPart> create()
    {
        return adoptRef(*new SwitchThumbPart(false, 0.0f));
    }

    static Ref<SwitchThumbPart> create(bool isOn, float progress)
    {
        return adoptRef(*new SwitchThumbPart(isOn, progress));
    }

    SwitchThumbPart(bool isOn, float progress)
        : ControlPart(StyleAppearance::SwitchThumb)
        , m_isOn(isOn)
        , m_progress(progress)
    {
    }

    bool isOn() const { return m_isOn; }
    void setIsOn(bool isOn) { m_isOn = isOn; }

    float progress() const { return m_progress; }
    void setProgress(float progress) { m_progress = progress; }

private:
    SwitchThumbPart()
        : ControlPart(StyleAppearance::SwitchThumb)
    {
    }

    std::unique_ptr<PlatformControl> createPlatformControl() final
    {
        return controlFactory().createPlatformSwitchThumb(*this);
    }

    bool m_isOn;
    float m_progress;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_CONTROL_PART(SwitchThumb)
