/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 28, 2024.
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

#if USE(THEME_ADWAITA)

#include "ControlFactory.h"

namespace WebCore {

class ControlFactoryAdwaita final : public ControlFactory {
public:
    static ControlFactoryAdwaita& shared();

private:
    std::unique_ptr<PlatformControl> createPlatformButton(ButtonPart&) final;
    std::unique_ptr<PlatformControl> createPlatformColorWell(ColorWellPart&) final;
#if ENABLE(SERVICE_CONTROLS)
    std::unique_ptr<PlatformControl> createPlatformImageControlsButton(ImageControlsButtonPart&) final;
#endif
    std::unique_ptr<PlatformControl> createPlatformInnerSpinButton(InnerSpinButtonPart&) final;
    std::unique_ptr<PlatformControl> createPlatformMenuList(MenuListPart&) final;
    std::unique_ptr<PlatformControl> createPlatformMenuListButton(MenuListButtonPart&) final;
    std::unique_ptr<PlatformControl> createPlatformMeter(MeterPart&) final;
    std::unique_ptr<PlatformControl> createPlatformProgressBar(ProgressBarPart&) final;
    std::unique_ptr<PlatformControl> createPlatformSearchField(SearchFieldPart&) final;
    std::unique_ptr<PlatformControl> createPlatformSearchFieldCancelButton(SearchFieldCancelButtonPart&) final;
    std::unique_ptr<PlatformControl> createPlatformSearchFieldResults(SearchFieldResultsPart&) final;
    std::unique_ptr<PlatformControl> createPlatformSliderThumb(SliderThumbPart&) final;
    std::unique_ptr<PlatformControl> createPlatformSliderTrack(SliderTrackPart&) final;
    std::unique_ptr<PlatformControl> createPlatformSwitchThumb(SwitchThumbPart&) final;
    std::unique_ptr<PlatformControl> createPlatformSwitchTrack(SwitchTrackPart&) final;
    std::unique_ptr<PlatformControl> createPlatformTextArea(TextAreaPart&) final;
    std::unique_ptr<PlatformControl> createPlatformTextField(TextFieldPart&) final;
    std::unique_ptr<PlatformControl> createPlatformToggleButton(ToggleButtonPart&) final;
};

} // namespace WebCore

#endif // USE(THEME_ADWAITA)
