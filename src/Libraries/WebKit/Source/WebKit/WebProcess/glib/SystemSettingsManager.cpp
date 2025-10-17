/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 22, 2023.
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
#include "SystemSettingsManager.h"

#if PLATFORM(GTK) || PLATFORM(WPE)
#include "SystemSettingsManagerMessages.h"
#include "WebProcess.h"
#include "WebProcessCreationParameters.h"
#include <WebCore/FontRenderOptions.h>
#include <WebCore/Page.h>
#include <WebCore/RenderTheme.h>
#include <WebCore/ScrollbarTheme.h>

namespace WebKit {
using namespace WebCore;

WTF_MAKE_TZONE_ALLOCATED_IMPL(SystemSettingsManager);

SystemSettingsManager::SystemSettingsManager(WebProcess& process)
    : m_process(process)
{
    process.addMessageReceiver(Messages::SystemSettingsManager::messageReceiverName(), *this);

    SystemSettings::singleton().addObserver([](const SystemSettings::State& state) {
        auto& systemSettings = SystemSettings::singleton();
        auto& fontRenderOptions = FontRenderOptions::singleton();
        bool antialiasSettingsDidChange = state.xftAntialias || state.xftRGBA;
        bool hintingSettingsDidChange = state.xftHinting || state.xftHintStyle;
        bool themeDidChange = state.themeName || state.darkMode;

        if (themeDidChange)
            RenderTheme::singleton().platformColorsDidChange();

        if (hintingSettingsDidChange)
            fontRenderOptions.setHinting(systemSettings.hintStyle());

        if (antialiasSettingsDidChange) {
            fontRenderOptions.setSubpixelOrder(systemSettings.subpixelOrder());
            fontRenderOptions.setAntialias(systemSettings.antialiasMode());
        }

        if (state.followFontSystemSettings)
            fontRenderOptions.setFollowSystemSettings(systemSettings.followFontSystemSettings());

        if (state.overlayScrolling || state.themeName)
            ScrollbarTheme::theme().themeChanged();

        if (themeDidChange || antialiasSettingsDidChange || hintingSettingsDidChange || state.followFontSystemSettings)
            Page::updateStyleForAllPagesAfterGlobalChangeInEnvironment();
    }, this);
}

void SystemSettingsManager::ref() const
{
    m_process->ref();
}

void SystemSettingsManager::deref() const
{
    m_process->deref();
}

SystemSettingsManager::~SystemSettingsManager()
{
    SystemSettings::singleton().removeObserver(this);
}

ASCIILiteral SystemSettingsManager::supplementName()
{
    return "SystemSettingsManager"_s;
}

void SystemSettingsManager::initialize(const WebProcessCreationParameters& parameters)
{
    didChange(parameters.systemSettings);
}

void SystemSettingsManager::didChange(const SystemSettings::State& state)
{
    SystemSettings::singleton().updateSettings(state);
}

} // namespace WebKit

#endif // PLATFORM(GTK) || PLATFORM(WPE)
