/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 28, 2024.
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
#include "ManetteGamepadProvider.h"

#if ENABLE(GAMEPAD) && OS(LINUX)

#include "GUniquePtrManette.h"
#include "GamepadProviderClient.h"
#include "Logging.h"
#include "ManetteGamepad.h"
#include <wtf/NeverDestroyed.h>

namespace WebCore {

static const Seconds connectionDelayInterval { 500_ms };
static const Seconds inputNotificationDelay { 50_ms };

ManetteGamepadProvider& ManetteGamepadProvider::singleton()
{
    static NeverDestroyed<ManetteGamepadProvider> sharedProvider;
    return sharedProvider;
}

static void onDeviceConnected(ManetteMonitor*, ManetteDevice* device, ManetteGamepadProvider* provider)
{
    provider->deviceConnected(device);
}

static void onDeviceDisconnected(ManetteMonitor*, ManetteDevice* device, ManetteGamepadProvider* provider)
{
    provider->deviceDisconnected(device);
}

ManetteGamepadProvider::ManetteGamepadProvider()
    : m_monitor(adoptGRef(manette_monitor_new()))
    , m_initialGamepadsConnectedTimer(RunLoop::current(), this, &ManetteGamepadProvider::initialGamepadsConnectedTimerFired)
    , m_inputNotificationTimer(RunLoop::current(), this, &ManetteGamepadProvider::inputNotificationTimerFired)
{
    g_signal_connect(m_monitor.get(), "device-connected", G_CALLBACK(onDeviceConnected), this);
    g_signal_connect(m_monitor.get(), "device-disconnected", G_CALLBACK(onDeviceDisconnected), this);
}

ManetteGamepadProvider::~ManetteGamepadProvider()
{
    g_signal_handlers_disconnect_by_data(m_monitor.get(), this);
}

void ManetteGamepadProvider::startMonitoringGamepads(GamepadProviderClient& client)
{
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

    RunLoop::current().dispatch([this] {
        ManetteDevice* device;
        GUniquePtr<ManetteMonitorIter> iter(manette_monitor_iterate(m_monitor.get()));
        while (manette_monitor_iter_next(iter.get(), &device))
            deviceConnected(device);
    });
}

void ManetteGamepadProvider::stopMonitoringGamepads(GamepadProviderClient& client)
{
    ASSERT(m_clients.contains(client));

    bool shouldCloseAndUnscheduleManager = m_clients.remove(client) && m_clients.isEmptyIgnoringNullReferences();
    if (shouldCloseAndUnscheduleManager) {
        m_gamepadVector.clear();
        m_gamepadMap.clear();
        m_initialGamepadsConnectedTimer.stop();
    }
}

void ManetteGamepadProvider::gamepadHadInput(ManetteGamepad&, ShouldMakeGamepadsVisible shouldMakeGamepadsVisible)
{
    if (!m_inputNotificationTimer.isActive())
        m_inputNotificationTimer.startOneShot(inputNotificationDelay);

    if (shouldMakeGamepadsVisible == ShouldMakeGamepadsVisible::Yes)
        setShouldMakeGamepadsVisibile();
}

void ManetteGamepadProvider::deviceConnected(ManetteDevice* device)
{
    ASSERT(!m_gamepadMap.get(device));

    LOG(Gamepad, "ManetteGamepadProvider device %p added", device);

    unsigned index = indexForNewlyConnectedDevice();
    auto gamepad = makeUnique<ManetteGamepad>(device, index);

    if (m_gamepadVector.size() <= index)
        m_gamepadVector.grow(index + 1);

    m_gamepadVector[index] = gamepad.get();
    m_gamepadMap.set(device, WTFMove(gamepad));

    if (!m_initialGamepadsConnected) {
        // This added device is the result of us starting to monitor gamepads.
        // We'll get notified of all connected devices during this current spin of the runloop
        // and the client should be told they were already connected.
        // The m_initialGamepadsConnectedTimer fires in a subsequent spin of the runloop after which
        // any connection events are actual new devices and can trigger gamepad visibility.
        if (!m_initialGamepadsConnectedTimer.isActive())
            m_initialGamepadsConnectedTimer.startOneShot(0_s);
    }

    auto eventVisibility = m_initialGamepadsConnected ? EventMakesGamepadsVisible::Yes : EventMakesGamepadsVisible::No;
    for (auto& client : m_clients)
        client.platformGamepadConnected(*m_gamepadVector[index], eventVisibility);
}

void ManetteGamepadProvider::deviceDisconnected(ManetteDevice* device)
{
    LOG(Gamepad, "ManetteGamepadProvider device %p removed", device);

    std::unique_ptr<ManetteGamepad> removedGamepad = removeGamepadForDevice(device);
    ASSERT(removedGamepad);

    for (auto& client : m_clients)
        client.platformGamepadDisconnected(*removedGamepad);
}

unsigned ManetteGamepadProvider::indexForNewlyConnectedDevice()
{
    unsigned index = 0;
    while (index < m_gamepadVector.size() && m_gamepadVector[index])
        ++index;

    return index;
}

void ManetteGamepadProvider::initialGamepadsConnectedTimerFired()
{
    m_initialGamepadsConnected = true;
}

void ManetteGamepadProvider::inputNotificationTimerFired()
{
    if (!m_initialGamepadsConnected) {
        if (!m_inputNotificationTimer.isActive())
            m_inputNotificationTimer.startOneShot(0_s);
        return;
    }

    dispatchPlatformGamepadInputActivity();
}

std::unique_ptr<ManetteGamepad> ManetteGamepadProvider::removeGamepadForDevice(ManetteDevice* device)
{
    std::unique_ptr<ManetteGamepad> result = m_gamepadMap.take(device);
    ASSERT(result);

    auto index = m_gamepadVector.find(result.get());
    if (index != notFound)
        m_gamepadVector[index] = nullptr;

    return result;
}

void ManetteGamepadProvider::playEffect(unsigned, const String&, GamepadHapticEffectType, const GamepadEffectParameters&, CompletionHandler<void(bool)>&& completionHandler)
{
    // Not supported by this provider.
    completionHandler(false);
}

void ManetteGamepadProvider::stopEffects(unsigned, const String&, CompletionHandler<void()>&& completionHandler)
{
    // Not supported by this provider.
    completionHandler();
}

} // namespace WebCore

#endif // ENABLE(GAMEPAD) && OS(LINUX)
