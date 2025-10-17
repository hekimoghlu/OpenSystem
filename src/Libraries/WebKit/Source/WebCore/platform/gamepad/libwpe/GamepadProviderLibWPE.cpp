/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 18, 2022.
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
#include "GamepadProviderLibWPE.h"

#if ENABLE(GAMEPAD) && USE(LIBWPE)

#include "GamepadLibWPE.h"
#include "GamepadProviderClient.h"
#include "Logging.h"
#include <inttypes.h>
#include <wpe/wpe.h>
#include <wtf/NeverDestroyed.h>

namespace WebCore {

static const Seconds connectionDelayInterval { 500_ms };
static const Seconds inputNotificationDelay { 5_ms };

GamepadProviderLibWPE& GamepadProviderLibWPE::singleton()
{
    static NeverDestroyed<GamepadProviderLibWPE> sharedProvider;
    return sharedProvider;
}

GamepadProviderLibWPE::GamepadProviderLibWPE()
    : m_provider(wpe_gamepad_provider_create(), wpe_gamepad_provider_destroy)
    , m_initialGamepadsConnectedTimer(RunLoop::current(), this, &GamepadProviderLibWPE::initialGamepadsConnectedTimerFired)
    , m_inputNotificationTimer(RunLoop::current(), this, &GamepadProviderLibWPE::inputNotificationTimerFired)
{
    static const struct wpe_gamepad_provider_client_interface s_client = {
        // connected
        [](void* data, uintptr_t gamepadId) {
            auto& provider = *static_cast<GamepadProviderLibWPE*>(data);
            provider.gamepadConnected(gamepadId);
        },
        // disconnected
        [](void* data, uintptr_t gamepadId) {
            auto& provider = *static_cast<GamepadProviderLibWPE*>(data);
            provider.gamepadDisconnected(gamepadId);
        },
        nullptr, nullptr, nullptr,
    };

    wpe_gamepad_provider_set_client(m_provider.get(), &s_client, this);
}

GamepadProviderLibWPE::~GamepadProviderLibWPE()
{
    wpe_gamepad_provider_set_client(m_provider.get(), nullptr, nullptr);
}

void GamepadProviderLibWPE::startMonitoringGamepads(GamepadProviderClient& client)
{
    if (!m_provider)
        return;

    bool shouldOpenAndScheduleManager = m_clients.isEmptyIgnoringNullReferences();

    ASSERT(!m_clients.contains(client));
    m_clients.add(client);

    if (!shouldOpenAndScheduleManager)
        return;

    ASSERT(m_gamepadVector.isEmpty());
    ASSERT(m_gamepadMap.isEmpty());

    m_initialGamepadsConnected = false;

    // Any connections we are notified of within the connectionDelayInterval of listening likely represent
    // devices that were already connected, so we suppress notifying clients of these.
    m_initialGamepadsConnectedTimer.startOneShot(connectionDelayInterval);

    wpe_gamepad_provider_start(m_provider.get());
}

void GamepadProviderLibWPE::stopMonitoringGamepads(GamepadProviderClient& client)
{
    if (!m_provider)
        return;

    ASSERT(m_clients.contains(client));

    bool shouldCloseAndUnscheduleManager = m_clients.remove(client) && m_clients.isEmptyIgnoringNullReferences();
    if (!shouldCloseAndUnscheduleManager)
        return;

    wpe_gamepad_provider_stop(m_provider.get());

    m_gamepadVector.clear();
    m_gamepadMap.clear();
    m_initialGamepadsConnectedTimer.stop();
    m_lastActiveGamepad = nullptr;
}

void GamepadProviderLibWPE::gamepadConnected(uintptr_t id)
{
    ASSERT(!m_gamepadMap.get(id));
    ASSERT(m_provider);

    LOG(Gamepad, "GamepadProviderLibWPE device %" PRIuPTR " added", id);

    unsigned index = indexForNewlyConnectedDevice();
    auto gamepad = makeUnique<GamepadLibWPE>(m_provider.get(), id, index);

    if (m_gamepadVector.size() <= index)
        m_gamepadVector.grow(index + 1);

    m_gamepadVector[index] = gamepad.get();
    m_gamepadMap.set(id, WTFMove(gamepad));

    if (!m_initialGamepadsConnected) {
        // This added device is the result of us starting to monitor gamepads.
        // We'll get notified of all connected devices during this current spin of the runloop
        // and the client should be told they were already connected.
        // The m_connectionDelayTimer fires in a subsequent spin of the runloop after which
        // any connection events are actual new devices and can trigger gamepad visibility.
        if (!m_initialGamepadsConnectedTimer.isActive())
            m_initialGamepadsConnectedTimer.startOneShot(0_s);
    }

    auto eventVisibility = m_initialGamepadsConnected ? EventMakesGamepadsVisible::Yes : EventMakesGamepadsVisible::No;
    for (auto& client : m_clients)
        client.platformGamepadConnected(*m_gamepadVector[index], eventVisibility);
}

void GamepadProviderLibWPE::gamepadDisconnected(uintptr_t id)
{
    LOG(Gamepad, "GamepadProviderLibWPE device %" PRIuPTR " removed", id);

    auto removedGamepad = removeGamepadForId(id);
    ASSERT(removedGamepad);

    if (removedGamepad->wpeGamepad() == m_lastActiveGamepad)
        m_lastActiveGamepad = nullptr;

    for (auto& client : m_clients)
        client.platformGamepadDisconnected(*removedGamepad);
}

unsigned GamepadProviderLibWPE::indexForNewlyConnectedDevice()
{
    unsigned index = 0;
    while (index < m_gamepadVector.size() && m_gamepadVector[index])
        ++index;

    return index;
}

std::unique_ptr<GamepadLibWPE> GamepadProviderLibWPE::removeGamepadForId(uintptr_t id)
{
    auto removedGamepad = m_gamepadMap.take(id);
    ASSERT(removedGamepad);

    auto index = m_gamepadVector.find(removedGamepad.get());
    if (index != notFound)
        m_gamepadVector[index] = nullptr;

    if (removedGamepad->wpeGamepad() == m_lastActiveGamepad)
        m_lastActiveGamepad = nullptr;

    return removedGamepad;
}

void GamepadProviderLibWPE::initialGamepadsConnectedTimerFired()
{
    m_initialGamepadsConnected = true;
}

void GamepadProviderLibWPE::inputNotificationTimerFired()
{
    if (!m_initialGamepadsConnected) {
        if (!m_inputNotificationTimer.isActive())
            m_inputNotificationTimer.startOneShot(0_s);
        return;
    }

    dispatchPlatformGamepadInputActivity();
}

void GamepadProviderLibWPE::scheduleInputNotification(GamepadLibWPE& gamepad, ShouldMakeGamepadsVisible shouldMakeGamepadsVisible)
{
    m_lastActiveGamepad = const_cast<struct wpe_gamepad*>(gamepad.wpeGamepad());

    if (!m_inputNotificationTimer.isActive())
        m_inputNotificationTimer.startOneShot(inputNotificationDelay);

    if (shouldMakeGamepadsVisible == ShouldMakeGamepadsVisible::Yes)
        setShouldMakeGamepadsVisibile();
}

struct wpe_view_backend* GamepadProviderLibWPE::inputView()
{
    if (!m_provider || !m_lastActiveGamepad)
        return nullptr;
    return wpe_gamepad_provider_get_view_backend(m_provider.get(), m_lastActiveGamepad);
}

void GamepadProviderLibWPE::playEffect(unsigned, const String&, GamepadHapticEffectType, const GamepadEffectParameters&, CompletionHandler<void(bool)>&& completionHandler)
{
    // Not supported by this provider.
    completionHandler(false);
}

void GamepadProviderLibWPE::stopEffects(unsigned, const String&, CompletionHandler<void()>&& completionHandler)
{
    // Not supported by this provider.
    completionHandler();
}

} // namespace WebCore

#endif // ENABLE(GAMEPAD) && USE(LIBWPE)
