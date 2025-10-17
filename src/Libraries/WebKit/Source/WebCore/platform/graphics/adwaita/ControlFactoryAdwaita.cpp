/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 23, 2024.
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
#include "ControlFactoryAdwaita.h"

#include "ButtonControlAdwaita.h"
#include "ButtonPart.h"
#include "ColorWellPart.h"
#include "InnerSpinButtonAdwaita.h"
#include "InnerSpinButtonPart.h"
#include "MenuListAdwaita.h"
#include "MenuListButtonPart.h"
#include "MenuListPart.h"
#include "NotImplemented.h"
#include "ProgressBarAdwaita.h"
#include "SearchFieldPart.h"
#include "SliderThumbAdwaita.h"
#include "SliderThumbPart.h"
#include "SliderTrackAdwaita.h"
#include "TextAreaPart.h"
#include "TextFieldAdwaita.h"
#include "TextFieldPart.h"
#include "ToggleButtonAdwaita.h"
#include "ToggleButtonPart.h"


#if USE(THEME_ADWAITA)

namespace WebCore {
using namespace WebCore::Adwaita;

RefPtr<ControlFactory> ControlFactory::create()
{
    return adoptRef(new ControlFactoryAdwaita());
}

ControlFactoryAdwaita& ControlFactoryAdwaita::shared()
{
    return static_cast<ControlFactoryAdwaita&>(ControlFactory::shared());
}

#if ENABLE(APPLE_PAY)
std::unique_ptr<PlatformControl> ControlFactoryAdwaita::createPlatformApplePayButton(ApplePayButtonPart&)
{
    notImplemented();
    return nullptr;
}
#endif

std::unique_ptr<PlatformControl> ControlFactoryAdwaita::createPlatformButton(ButtonPart& part)
{
    return makeUnique<ButtonControlAdwaita>(part, *this);
}

std::unique_ptr<PlatformControl> ControlFactoryAdwaita::createPlatformColorWell(ColorWellPart& part)
{
    return makeUnique<ButtonControlAdwaita>(part, *this);
}

#if ENABLE(SERVICE_CONTROLS)
std::unique_ptr<PlatformControl> ControlFactoryAdwaita::createPlatformImageControlsButton(ImageControlsButtonPart&)
{
    notImplemented();
    return nullptr;
}
#endif

std::unique_ptr<PlatformControl> ControlFactoryAdwaita::createPlatformInnerSpinButton(InnerSpinButtonPart& part)
{
    return makeUnique<InnerSpinButtonAdwaita>(part, *this);
}

std::unique_ptr<PlatformControl> ControlFactoryAdwaita::createPlatformMenuList(MenuListPart& part)
{
    return makeUnique<MenuListAdwaita>(part, *this);
}

std::unique_ptr<PlatformControl> ControlFactoryAdwaita::createPlatformMenuListButton(MenuListButtonPart& part)
{
    return makeUnique<MenuListAdwaita>(part, *this);
}

std::unique_ptr<PlatformControl> ControlFactoryAdwaita::createPlatformMeter(MeterPart&)
{
    notImplemented();
    return nullptr;
}

std::unique_ptr<PlatformControl> ControlFactoryAdwaita::createPlatformProgressBar(ProgressBarPart& part)
{
    return makeUnique<ProgressBarAdwaita>(part, *this);
}

std::unique_ptr<PlatformControl> ControlFactoryAdwaita::createPlatformSearchField(SearchFieldPart& part)
{
    return makeUnique<TextFieldAdwaita>(part, *this);
}

std::unique_ptr<PlatformControl> ControlFactoryAdwaita::createPlatformSearchFieldCancelButton(SearchFieldCancelButtonPart&)
{
    notImplemented();
    return nullptr;
}

std::unique_ptr<PlatformControl> ControlFactoryAdwaita::createPlatformSearchFieldResults(SearchFieldResultsPart&)
{
    notImplemented();
    return nullptr;
}

std::unique_ptr<PlatformControl> ControlFactoryAdwaita::createPlatformSliderThumb(SliderThumbPart& part)
{
    return makeUnique<SliderThumbAdwaita>(part, *this);
}

std::unique_ptr<PlatformControl> ControlFactoryAdwaita::createPlatformSliderTrack(SliderTrackPart& part)
{
    return makeUnique<SliderTrackAdwaita>(part, *this);
}

std::unique_ptr<PlatformControl> ControlFactoryAdwaita::createPlatformSwitchThumb(SwitchThumbPart&)
{
    notImplemented();
    return nullptr;
}

std::unique_ptr<PlatformControl> ControlFactoryAdwaita::createPlatformSwitchTrack(SwitchTrackPart&)
{
    notImplemented();
    return nullptr;
}

std::unique_ptr<PlatformControl> ControlFactoryAdwaita::createPlatformTextArea(TextAreaPart& part)
{
    return makeUnique<TextFieldAdwaita>(part, *this);
}

std::unique_ptr<PlatformControl> ControlFactoryAdwaita::createPlatformTextField(TextFieldPart& part)
{
    return makeUnique<TextFieldAdwaita>(part, *this);
}

std::unique_ptr<PlatformControl> ControlFactoryAdwaita::createPlatformToggleButton(ToggleButtonPart& part)
{
    return makeUnique<ToggleButtonAdwaita>(part, *this);
}

} // namespace WebCore

#endif // USE(THEME_ADWAITA)
