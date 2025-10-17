/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 26, 2025.
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

#if PLATFORM(MAC)

#import "ControlFactoryCocoa.h"
#import "WebControlView.h"
#import <wtf/TZoneMalloc.h>

OBJC_CLASS NSServicesRolloverButtonCell;

namespace WebCore {

class FloatRect;
struct ControlStyle;

class ControlFactoryMac final : public ControlFactoryCocoa {
    WTF_MAKE_TZONE_ALLOCATED(ControlFactoryMac);
public:
    using ControlFactoryCocoa::ControlFactoryCocoa;

    static ControlFactoryMac& shared();

    NSView *drawingView(const FloatRect&, const ControlStyle&) const;

#if ENABLE(SERVICE_CONTROLS)
    NSServicesRolloverButtonCell *servicesRolloverButtonCell() const;
#endif

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

    NSButtonCell *buttonCell() const;
    NSButtonCell *defaultButtonCell() const;
    NSButtonCell *checkboxCell() const;
    NSButtonCell *radioCell() const;
    NSLevelIndicatorCell *levelIndicatorCell() const;
    NSPopUpButtonCell *popUpButtonCell() const;
    NSSearchFieldCell *searchFieldCell() const;
    NSMenu *searchMenuTemplate() const;
    NSSliderCell *sliderCell() const;
    NSStepperCell *stepperCell() const;
    NSTextFieldCell *textFieldCell() const;

    mutable RetainPtr<WebControlView> m_drawingView;

    mutable RetainPtr<NSButtonCell> m_buttonCell;
    mutable RetainPtr<NSButtonCell> m_defaultButtonCell;
    mutable RetainPtr<NSButtonCell> m_checkboxCell;
    mutable RetainPtr<NSButtonCell> m_radioCell;
    mutable RetainPtr<NSLevelIndicatorCell> m_levelIndicatorCell;
    mutable RetainPtr<NSPopUpButtonCell> m_popUpButtonCell;
#if ENABLE(SERVICE_CONTROLS)
    mutable RetainPtr<NSServicesRolloverButtonCell> m_servicesRolloverButtonCell;
#endif
    mutable RetainPtr<NSSearchFieldCell> m_searchFieldCell;
    mutable RetainPtr<NSMenu> m_searchMenuTemplate;
    mutable RetainPtr<NSSliderCell> m_sliderCell;
    mutable RetainPtr<NSStepperCell> m_stepperCell;
    mutable RetainPtr<NSTextFieldCell> m_textFieldCell;
};

} // namespace WebCore

#endif // PLATFORM(MAC)
