/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 13, 2022.
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
#include "MultiGamepadProvider.h"

#if ENABLE(GAMEPAD) && HAVE(MULTIGAMEPADPROVIDER_SUPPORT)

#import "GameControllerGamepadProvider.h"
#import "HIDGamepadProvider.h"
#import "Logging.h"
#import "PlatformGamepad.h"
#import <wtf/NeverDestroyed.h>

namespace WebCore {

static size_t numberOfGamepadProviders = 2;

MultiGamepadProvider& MultiGamepadProvider::singleton()
{
    static NeverDestroyed<MultiGamepadProvider> sharedProvider;
    return sharedProvider;
}

void MultiGamepadProvider::startMonitoringGamepads(GamepadProviderClient& client)
{
    bool monitorOtherProviders = m_clients.isEmptyIgnoringNullReferences();

    ASSERT(!m_clients.contains(client));
    m_clients.add(client);

    if (!m_usesOnlyHIDProvider) {
        HIDGamepadProvider::singleton().ignoreGameControllerFrameworkDevices();
        GameControllerGamepadProvider::singleton().prewarmGameControllerDevicesIfNecessary();
    }

    if (monitorOtherProviders) {
        HIDGamepadProvider::singleton().startMonitoringGamepads(*this);
        if (!m_usesOnlyHIDProvider)
            GameControllerGamepadProvider::singleton().startMonitoringGamepads(*this);
    }
}

void MultiGamepadProvider::stopMonitoringGamepads(GamepadProviderClient& client)
{
    ASSERT(m_clients.contains(client));

    bool shouldStopMonitoringOtherProviders = m_clients.remove(client) && m_clients.isEmptyIgnoringNullReferences();

    if (shouldStopMonitoringOtherProviders) {
        HIDGamepadProvider::singleton().stopMonitoringGamepads(*this);
        if (!m_usesOnlyHIDProvider)
            GameControllerGamepadProvider::singleton().stopMonitoringGamepads(*this);
    }
}

void MultiGamepadProvider::playEffect(unsigned, const String&, GamepadHapticEffectType, const GamepadEffectParameters&, CompletionHandler<void(bool)>&& completionHandler)
{
    // Not supported by this provider.
    completionHandler(false);
}

void MultiGamepadProvider::stopEffects(unsigned, const String&, CompletionHandler<void()>&& completionHandler)
{
    // Not supported by this provider.
    completionHandler();
}

unsigned MultiGamepadProvider::indexForNewlyConnectedDevice()
{
    unsigned index = 0;
    while (index < m_gamepadVector.size() && m_gamepadVector[index])
        ++index;

    ASSERT(index <= m_gamepadVector.size());

    if (index == m_gamepadVector.size())
        m_gamepadVector.resize(index + 1);

    return index;
}

void MultiGamepadProvider::platformGamepadConnected(PlatformGamepad& gamepad, EventMakesGamepadsVisible eventVisibility)
{
    auto index = indexForNewlyConnectedDevice();

    LOG(Gamepad, "MultiGamepadProvider adding new platform gamepad to index %i from a %s source", index, gamepad.source());

    ASSERT(m_gamepadVector.size() > index);

    auto addResult = m_gamepadMap.add(gamepad, WTF::makeUnique<PlatformGamepadWrapper>(index, &gamepad));
    ASSERT(addResult.isNewEntry);
    m_gamepadVector[index] = addResult.iterator->value.get();

    for (auto& client : m_clients)
        client.platformGamepadConnected(*m_gamepadVector[index], eventVisibility);
}

void MultiGamepadProvider::platformGamepadDisconnected(PlatformGamepad& gamepad)
{
    LOG(Gamepad, "MultiGamepadProvider disconnecting gamepad from a %s source", gamepad.source());

    auto gamepadWrapper = m_gamepadMap.take(gamepad);

    ASSERT(gamepadWrapper);
    ASSERT(gamepadWrapper->index() < m_gamepadVector.size());
    ASSERT(m_gamepadVector[gamepadWrapper->index()] == gamepadWrapper.get());

    m_gamepadVector[gamepadWrapper->index()] = nullptr;

    for (auto& client : m_clients)
        client.platformGamepadDisconnected(*gamepadWrapper);
}

void MultiGamepadProvider::platformGamepadInputActivity(EventMakesGamepadsVisible eventVisibility)
{
    if (eventVisibility == EventMakesGamepadsVisible::Yes)
        GameControllerGamepadProvider::singleton().makeInvisibleGamepadsVisible();

    for (auto& client : m_clients)
        client.platformGamepadInputActivity(eventVisibility);
}

} // namespace WebCore

#endif // ENABLE(GAMEPAD) && HAVE(MULTIGAMEPADPROVIDER_SUPPORT)
