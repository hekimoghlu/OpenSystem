/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 10, 2024.
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

#include <wtf/RefCounted.h>
#include <wtf/RefPtr.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class ApplePayButtonPart;
class ButtonPart;
class ColorWellPart;
class ImageControlsButtonPart;
class InnerSpinButtonPart;
class MeterPart;
class MenuListButtonPart;
class MenuListPart;
class PlatformControl;
class ProgressBarPart;
class SearchFieldCancelButtonPart;
class SearchFieldPart;
class SearchFieldResultsPart;
class SliderThumbPart;
class SliderTrackPart;
class SwitchThumbPart;
class SwitchTrackPart;
class TextAreaPart;
class TextFieldPart;
class ToggleButtonPart;

class ControlFactory : public RefCounted<ControlFactory> {
    WTF_MAKE_TZONE_ALLOCATED(ControlFactory);
public:
    virtual ~ControlFactory() = default;

    WEBCORE_EXPORT static RefPtr<ControlFactory> create();
    WEBCORE_EXPORT static ControlFactory& shared();

#if ENABLE(APPLE_PAY)
    virtual std::unique_ptr<PlatformControl> createPlatformApplePayButton(ApplePayButtonPart&) = 0;
#endif
    virtual std::unique_ptr<PlatformControl> createPlatformButton(ButtonPart&) = 0;
    virtual std::unique_ptr<PlatformControl> createPlatformColorWell(ColorWellPart&) = 0;
#if ENABLE(SERVICE_CONTROLS)
    virtual std::unique_ptr<PlatformControl> createPlatformImageControlsButton(ImageControlsButtonPart&) = 0;
#endif
    virtual std::unique_ptr<PlatformControl> createPlatformInnerSpinButton(InnerSpinButtonPart&) = 0;
    virtual std::unique_ptr<PlatformControl> createPlatformMenuList(MenuListPart&) = 0;
    virtual std::unique_ptr<PlatformControl> createPlatformMenuListButton(MenuListButtonPart&) = 0;
    virtual std::unique_ptr<PlatformControl> createPlatformMeter(MeterPart&) = 0;
    virtual std::unique_ptr<PlatformControl> createPlatformProgressBar(ProgressBarPart&) = 0;
    virtual std::unique_ptr<PlatformControl> createPlatformSearchField(SearchFieldPart&) = 0;
    virtual std::unique_ptr<PlatformControl> createPlatformSearchFieldCancelButton(SearchFieldCancelButtonPart&) = 0;
    virtual std::unique_ptr<PlatformControl> createPlatformSearchFieldResults(SearchFieldResultsPart&) = 0;
    virtual std::unique_ptr<PlatformControl> createPlatformSliderThumb(SliderThumbPart&) = 0;
    virtual std::unique_ptr<PlatformControl> createPlatformSliderTrack(SliderTrackPart&) = 0;
    virtual std::unique_ptr<PlatformControl> createPlatformSwitchThumb(SwitchThumbPart&) = 0;
    virtual std::unique_ptr<PlatformControl> createPlatformSwitchTrack(SwitchTrackPart&) = 0;
    virtual std::unique_ptr<PlatformControl> createPlatformTextArea(TextAreaPart&) = 0;
    virtual std::unique_ptr<PlatformControl> createPlatformTextField(TextFieldPart&) = 0;
    virtual std::unique_ptr<PlatformControl> createPlatformToggleButton(ToggleButtonPart&) = 0;

protected:
    ControlFactory() = default;
};

} // namespace WebCore
