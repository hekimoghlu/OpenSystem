/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 7, 2022.
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
#include "EmptyControlFactory.h"

#include "NotImplemented.h"
#include "PlatformControl.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(EmptyControlFactory);

#if ENABLE(APPLE_PAY)
std::unique_ptr<PlatformControl> EmptyControlFactory::createPlatformApplePayButton(ApplePayButtonPart&)
{
    notImplemented();
    return nullptr;
}
#endif

std::unique_ptr<PlatformControl> EmptyControlFactory::createPlatformButton(ButtonPart&)
{
    notImplemented();
    return nullptr;
}

std::unique_ptr<PlatformControl> EmptyControlFactory::createPlatformColorWell(ColorWellPart&)
{
    notImplemented();
    return nullptr;
}

#if ENABLE(SERVICE_CONTROLS)
std::unique_ptr<PlatformControl> EmptyControlFactory::createPlatformImageControlsButton(ImageControlsButtonPart&)
{
    notImplemented();
    return nullptr;
}
#endif

std::unique_ptr<PlatformControl> EmptyControlFactory::createPlatformInnerSpinButton(InnerSpinButtonPart&)
{
    notImplemented();
    return nullptr;
}

std::unique_ptr<PlatformControl> EmptyControlFactory::createPlatformMenuList(MenuListPart&)
{
    notImplemented();
    return nullptr;
}

std::unique_ptr<PlatformControl> EmptyControlFactory::createPlatformMenuListButton(MenuListButtonPart&)
{
    notImplemented();
    return nullptr;
}

std::unique_ptr<PlatformControl> EmptyControlFactory::createPlatformMeter(MeterPart&)
{
    notImplemented();
    return nullptr;
}

std::unique_ptr<PlatformControl> EmptyControlFactory::createPlatformProgressBar(ProgressBarPart&)
{
    notImplemented();
    return nullptr;
}

std::unique_ptr<PlatformControl> EmptyControlFactory::createPlatformSearchField(SearchFieldPart&)
{
    notImplemented();
    return nullptr;
}

std::unique_ptr<PlatformControl> EmptyControlFactory::createPlatformSearchFieldCancelButton(SearchFieldCancelButtonPart&)
{
    notImplemented();
    return nullptr;
}

std::unique_ptr<PlatformControl> EmptyControlFactory::createPlatformSearchFieldResults(SearchFieldResultsPart&)
{
    notImplemented();
    return nullptr;
}

std::unique_ptr<PlatformControl> EmptyControlFactory::createPlatformSliderThumb(SliderThumbPart&)
{
    notImplemented();
    return nullptr;
}

std::unique_ptr<PlatformControl> EmptyControlFactory::createPlatformSliderTrack(SliderTrackPart&)
{
    notImplemented();
    return nullptr;
}

std::unique_ptr<PlatformControl> EmptyControlFactory::createPlatformSwitchThumb(SwitchThumbPart&)
{
    notImplemented();
    return nullptr;
}

std::unique_ptr<PlatformControl> EmptyControlFactory::createPlatformSwitchTrack(SwitchTrackPart&)
{
    notImplemented();
    return nullptr;
}

std::unique_ptr<PlatformControl> EmptyControlFactory::createPlatformTextArea(TextAreaPart&)
{
    notImplemented();
    return nullptr;
}

std::unique_ptr<PlatformControl> EmptyControlFactory::createPlatformTextField(TextFieldPart&)
{
    notImplemented();
    return nullptr;
}

std::unique_ptr<PlatformControl> EmptyControlFactory::createPlatformToggleButton(ToggleButtonPart&)
{
    notImplemented();
    return nullptr;
}

} // namespace WebCore
