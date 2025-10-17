/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 26, 2023.
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
#import "config.h"
#import "ControlFactoryIOS.h"

#if PLATFORM(IOS_FAMILY)

#import "NotImplemented.h"
#import <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(ControlFactoryIOS);

RefPtr<ControlFactory> ControlFactory::create()
{
    return adoptRef(new ControlFactoryIOS());
}

std::unique_ptr<PlatformControl> ControlFactoryIOS::createPlatformButton(ButtonPart&)
{
    notImplemented();
    return nullptr;
}

std::unique_ptr<PlatformControl> ControlFactoryIOS::createPlatformColorWell(ColorWellPart&)
{
    notImplemented();
    return nullptr;
}

std::unique_ptr<PlatformControl> ControlFactoryIOS::createPlatformInnerSpinButton(InnerSpinButtonPart&)
{
    notImplemented();
    return nullptr;
}

std::unique_ptr<PlatformControl> ControlFactoryIOS::createPlatformMenuList(MenuListPart&)
{
    notImplemented();
    return nullptr;
}

std::unique_ptr<PlatformControl> ControlFactoryIOS::createPlatformMenuListButton(MenuListButtonPart&)
{
    notImplemented();
    return nullptr;
}

std::unique_ptr<PlatformControl> ControlFactoryIOS::createPlatformMeter(MeterPart&)
{
    notImplemented();
    return nullptr;
}

std::unique_ptr<PlatformControl> ControlFactoryIOS::createPlatformProgressBar(ProgressBarPart&)
{
    notImplemented();
    return nullptr;
}

std::unique_ptr<PlatformControl> ControlFactoryIOS::createPlatformSearchField(SearchFieldPart&)
{
    notImplemented();
    return nullptr;
}

std::unique_ptr<PlatformControl> ControlFactoryIOS::createPlatformSearchFieldCancelButton(SearchFieldCancelButtonPart&)
{
    notImplemented();
    return nullptr;
}

std::unique_ptr<PlatformControl> ControlFactoryIOS::createPlatformSearchFieldResults(SearchFieldResultsPart&)
{
    notImplemented();
    return nullptr;
}

std::unique_ptr<PlatformControl> ControlFactoryIOS::createPlatformSliderThumb(SliderThumbPart&)
{
    notImplemented();
    return nullptr;
}

std::unique_ptr<PlatformControl> ControlFactoryIOS::createPlatformSliderTrack(SliderTrackPart&)
{
    notImplemented();
    return nullptr;
}

std::unique_ptr<PlatformControl> ControlFactoryIOS::createPlatformSwitchThumb(SwitchThumbPart&)
{
    notImplemented();
    return nullptr;
}

std::unique_ptr<PlatformControl> ControlFactoryIOS::createPlatformSwitchTrack(SwitchTrackPart&)
{
    notImplemented();
    return nullptr;
}

std::unique_ptr<PlatformControl> ControlFactoryIOS::createPlatformTextArea(TextAreaPart&)
{
    notImplemented();
    return nullptr;
}

std::unique_ptr<PlatformControl> ControlFactoryIOS::createPlatformTextField(TextFieldPart&)
{
    notImplemented();
    return nullptr;
}

std::unique_ptr<PlatformControl> ControlFactoryIOS::createPlatformToggleButton(ToggleButtonPart&)
{
    notImplemented();
    return nullptr;
}

} // namespace WebCore

#endif // PLATFORM(IOS_FAMILY)
