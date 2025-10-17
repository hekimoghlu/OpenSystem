/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 24, 2022.
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
#include "MockGamepadProvider.h"

#if ENABLE(GAMEPAD)

#include "GamepadProviderClient.h"
#include "MockGamepad.h"
#include <wtf/CompletionHandler.h>
#include <wtf/MainThread.h>
#include <wtf/NeverDestroyed.h>

namespace WebCore {

MockGamepadProvider& MockGamepadProvider::singleton()
{
    static NeverDestroyed<MockGamepadProvider> sharedProvider;
    return sharedProvider;
}

MockGamepadProvider::MockGamepadProvider() = default;

void MockGamepadProvider::startMonitoringGamepads(GamepadProviderClient& client)
{
    ASSERT(!m_clients.contains(client));
    m_clients.add(client);
    WeakHashSet<PlatformGamepad> invisibleGamepads;
    for (auto& gamepad : m_connectedGamepadVector) {
        if (gamepad)
            invisibleGamepads.add(*gamepad);
    }
    ASSERT(!m_invisibleGamepadsForClient.contains(client));
    m_invisibleGamepadsForClient.set(client, invisibleGamepads);
}

void MockGamepadProvider::stopMonitoringGamepads(GamepadProviderClient& client)
{
    ASSERT(m_clients.contains(client));
    m_clients.remove(client);
    m_invisibleGamepadsForClient.remove(client);
}

void MockGamepadProvider::setMockGamepadDetails(unsigned index, const String& gamepadID, const String& mapping, unsigned axisCount, unsigned buttonCount, bool supportsDualRumble)
{
    if (index >= m_mockGamepadVector.size())
        m_mockGamepadVector.resize(index + 1);

    if (m_mockGamepadVector[index])
        m_mockGamepadVector[index]->updateDetails(gamepadID, mapping, axisCount, buttonCount, supportsDualRumble);
    else
        m_mockGamepadVector[index] = makeUnique<MockGamepad>(index, gamepadID, mapping, axisCount, buttonCount, supportsDualRumble);
}

bool MockGamepadProvider::connectMockGamepad(unsigned index)
{
    if (index < m_connectedGamepadVector.size() && m_connectedGamepadVector[index]) {
        LOG_ERROR("MockGamepadProvider: Attempt to connect a fake gamepad that is already connected (%u)", index);
        return false;
    }

    if (index >= m_mockGamepadVector.size() || !m_mockGamepadVector[index]) {
        LOG_ERROR("MockGamepadProvider: Attempt to connect a fake gamepad that doesn't have details set(%u)", index);
        return false;
    }

    if (m_connectedGamepadVector.size() <= index)
        m_connectedGamepadVector.reserveCapacity(index + 1);

    while (m_connectedGamepadVector.size() <= index)
        m_connectedGamepadVector.append(nullptr);

    m_connectedGamepadVector[index] = m_mockGamepadVector[index].get();

    for (auto& client : m_clients) {
        client.platformGamepadConnected(*m_connectedGamepadVector[index], EventMakesGamepadsVisible::Yes);
        auto gamepadsForClient = m_invisibleGamepadsForClient.find(client);
        if (gamepadsForClient != m_invisibleGamepadsForClient.end()) {
            for (auto& invisibleGamepad : gamepadsForClient->value)
                client.platformGamepadConnected(invisibleGamepad, EventMakesGamepadsVisible::Yes);
            m_invisibleGamepadsForClient.remove(gamepadsForClient);
        }
    }

    return true;
}

bool MockGamepadProvider::disconnectMockGamepad(unsigned index)
{
    if (index >= m_connectedGamepadVector.size() || !m_connectedGamepadVector[index]) {
        LOG_ERROR("MockGamepadProvider: Attempt to disconnect a fake gamepad that is not connected (%u)", index);
        return false;
    }
    if (m_connectedGamepadVector[index] != m_mockGamepadVector[index].get()) {
        LOG_ERROR("MockGamepadProvider: Vectors of fake gamepads and connected gamepads have gotten out of sync");
        return false;
    }

    m_connectedGamepadVector[index] = nullptr;

    auto gamepadToRemove = m_mockGamepadVector[index].get();
    for (auto& client : m_clients) {
        auto gamepadsForClient = m_invisibleGamepadsForClient.find(client);
        if (gamepadsForClient == m_invisibleGamepadsForClient.end() || !gamepadsForClient->value.remove(*gamepadToRemove))
            client.platformGamepadDisconnected(*m_mockGamepadVector[index]);
    }

    return true;
}

bool MockGamepadProvider::setMockGamepadAxisValue(unsigned index, unsigned axisIndex, double value)
{
    if (index >= m_mockGamepadVector.size() || !m_mockGamepadVector[index]) {
        LOG_ERROR("MockGamepadProvider: Attempt to set axis value on a fake gamepad that doesn't exist (%u)", index);
        return false;
    }

    m_mockGamepadVector[index]->setAxisValue(axisIndex, value);
    gamepadInputActivity();
    return true;
}

bool MockGamepadProvider::setMockGamepadButtonValue(unsigned index, unsigned buttonIndex, double value)
{
    if (index >= m_mockGamepadVector.size() || !m_mockGamepadVector[index]) {
        LOG_ERROR("MockGamepadProvider: Attempt to set button value on a fake gamepad that doesn't exist (%u)", index);
        return false;
    }

    m_mockGamepadVector[index]->setButtonValue(buttonIndex, value);
    setShouldMakeGamepadsVisibile();
    gamepadInputActivity();
    return true;
}

void MockGamepadProvider::gamepadInputActivity()
{
    if (!m_shouldScheduleActivityCallback)
        return;

    m_shouldScheduleActivityCallback = false;
    callOnMainThread([this]() {
        dispatchPlatformGamepadInputActivity();

        m_shouldScheduleActivityCallback = true;
    });
}

void MockGamepadProvider::clearGamepadsForTesting()
{
    // Disconnect any remaining connected gamepads.
    for (size_t i = 0; i < m_connectedGamepadVector.size(); ++i) {
        if (m_connectedGamepadVector[i])
            disconnectMockGamepad(i);
    }
    m_mockGamepadVector.clear();
}

void MockGamepadProvider::playEffect(unsigned, const String&, GamepadHapticEffectType, const GamepadEffectParameters&, CompletionHandler<void(bool)>&& completionHandler)
{
    completionHandler(true);
}

void MockGamepadProvider::stopEffects(unsigned, const String&, CompletionHandler<void()>&& completionHandler)
{
    completionHandler();
}

} // namespace WebCore

#endif // ENABLE(GAMEPAD)
