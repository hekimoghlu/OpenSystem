/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 24, 2023.
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
#include "SystemSettingsManagerProxy.h"

#include "SystemSettingsManagerMessages.h"
#include "WebProcessPool.h"
#include <WebCore/SystemSettings.h>

namespace WebKit {
using namespace WebCore;

#if !PLATFORM(GTK) && (!PLATFORM(WPE) || !ENABLE(WPE_PLATFORM))

SystemSettingsManagerProxy::SystemSettingsManagerProxy() = default;

String SystemSettingsManagerProxy::themeName() const
{
    return emptyString();
}

bool SystemSettingsManagerProxy::darkMode() const
{
    return false;
}

String SystemSettingsManagerProxy::fontName() const
{
    return "Sans 10"_s;
}

int SystemSettingsManagerProxy::xftAntialias() const
{
    return -1;
}

int SystemSettingsManagerProxy::xftHinting() const
{
    return -1;
}

String SystemSettingsManagerProxy::xftHintStyle() const
{
    return emptyString();
}

String SystemSettingsManagerProxy::xftRGBA() const
{
    return "rgb"_s;
}

int SystemSettingsManagerProxy::xftDPI() const
{
    return -1;
}

bool SystemSettingsManagerProxy::followFontSystemSettings() const
{
    return false;
}

bool SystemSettingsManagerProxy::cursorBlink() const
{
    return true;
}

int SystemSettingsManagerProxy::cursorBlinkTime() const
{
    return 1200;
}

bool SystemSettingsManagerProxy::primaryButtonWarpsSlider() const
{
    return true;
}

bool SystemSettingsManagerProxy::overlayScrolling() const
{
    return true;
}

bool SystemSettingsManagerProxy::enableAnimations() const
{
    return true;
}

#endif // !PLATFORM(GTK) && (!PLATFORM(WPE) || !ENABLE(WPE_PLATFORM))

void SystemSettingsManagerProxy::initialize()
{
    static NeverDestroyed<SystemSettingsManagerProxy> manager;
}

void SystemSettingsManagerProxy::settingsDidChange()
{
    const auto& oldState = SystemSettings::singleton().settingsState();
    SystemSettings::State changedState;

    auto themeName = this->themeName();
    if (oldState.themeName != themeName)
        changedState.themeName = themeName;

    auto darkMode = this->darkMode();
    if (oldState.darkMode != darkMode)
        changedState.darkMode = darkMode;

    auto fontName = this->fontName();
    if (oldState.fontName != fontName)
        changedState.fontName = fontName;

    auto xftAntialias = this->xftAntialias();
    if (xftAntialias != -1 && oldState.xftAntialias != xftAntialias)
        changedState.xftAntialias = xftAntialias;

    auto xftHinting = this->xftHinting();
    if (xftHinting != -1 && oldState.xftHinting != xftHinting)
        changedState.xftHinting = xftHinting;

    auto xftDPI = this->xftDPI();
    if (xftDPI != -1 && oldState.xftDPI != xftDPI)
        changedState.xftDPI = xftDPI;

    auto xftHintStyle = this->xftHintStyle();
    if (oldState.xftHintStyle != xftHintStyle)
        changedState.xftHintStyle = xftHintStyle;

    auto xftRGBA = this->xftRGBA();
    if (oldState.xftRGBA != xftRGBA)
        changedState.xftRGBA = xftRGBA;

    auto followFontSystemSettings = this->followFontSystemSettings();
    if (oldState.followFontSystemSettings != followFontSystemSettings)
        changedState.followFontSystemSettings = followFontSystemSettings;

    auto cursorBlink = this->cursorBlink();
    if (oldState.cursorBlink != cursorBlink)
        changedState.cursorBlink = cursorBlink;

    auto cursorBlinkTime = this->cursorBlinkTime();
    if (oldState.cursorBlinkTime != cursorBlinkTime)
        changedState.cursorBlinkTime = cursorBlinkTime;

    auto primaryButtonWarpsSlider = this->primaryButtonWarpsSlider();
    if (oldState.primaryButtonWarpsSlider != primaryButtonWarpsSlider)
        changedState.primaryButtonWarpsSlider = primaryButtonWarpsSlider;

    auto overlayScrolling = this->overlayScrolling();
    if (oldState.overlayScrolling != overlayScrolling)
        changedState.overlayScrolling = overlayScrolling;

    auto enableAnimations = this->enableAnimations();
    if (oldState.enableAnimations != enableAnimations)
        changedState.enableAnimations = enableAnimations;

    for (auto& processPool : WebProcessPool::allProcessPools())
        processPool->sendToAllProcesses(Messages::SystemSettingsManager::DidChange(changedState));

    SystemSettings::singleton().updateSettings(changedState);
}

} // namespace WebKit
