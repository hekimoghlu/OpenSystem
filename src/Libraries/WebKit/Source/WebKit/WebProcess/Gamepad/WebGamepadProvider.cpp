/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 23, 2024.
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
#include "WebGamepadProvider.h"

#if ENABLE(GAMEPAD)

#include "GamepadData.h"
#include "Logging.h"
#include "MessageSenderInlines.h"
#include "WebGamepad.h"
#include "WebProcess.h"
#include "WebProcessPoolMessages.h"
#include <WebCore/GamepadProviderClient.h>
#include <wtf/NeverDestroyed.h>

namespace WebKit {
using namespace WebCore;

#define WP_MESSAGE_CHECK(assertion, ...) { \
    if (UNLIKELY(!(assertion))) { \
        RELEASE_LOG_FAULT(IPC, "Exiting: %" PUBLIC_LOG_STRING " is false", #assertion); \
        CRASH_WITH_INFO(__VA_ARGS__); \
    } \
}

WebGamepadProvider& WebGamepadProvider::singleton()
{
    static NeverDestroyed<WebGamepadProvider> provider;
    return provider;
}

WebGamepadProvider::WebGamepadProvider()
{
}

WebGamepadProvider::~WebGamepadProvider()
{
}

void WebGamepadProvider::setInitialGamepads(const Vector<std::optional<GamepadData>>& gamepadDatas)
{
    WP_MESSAGE_CHECK((m_gamepads.isEmpty()), m_gamepads.size());

    m_gamepads.resize(gamepadDatas.size());
    m_rawGamepads.resize(gamepadDatas.size());
    for (size_t i = 0; i < gamepadDatas.size(); ++i) {
        if (!gamepadDatas[i])
            return;

        m_gamepads[i] = makeUnique<WebGamepad>(*gamepadDatas[i]);
        m_rawGamepads[i] = m_gamepads[i].get();
    }
}

void WebGamepadProvider::gamepadConnected(const GamepadData& gamepadData, EventMakesGamepadsVisible eventVisibility)
{
    LOG(Gamepad, "WebGamepadProvider::gamepadConnected - Gamepad index %u attached (visibility: %i)\n", gamepadData.index(), (int)eventVisibility);

    auto oldGamepadsSize = m_gamepads.size();
    if (m_gamepads.size() <= gamepadData.index()) {
        m_gamepads.grow(gamepadData.index() + 1);
        m_rawGamepads.grow(gamepadData.index() + 1);
    }
    WP_MESSAGE_CHECK((!m_gamepads[gamepadData.index()]), oldGamepadsSize, gamepadData.index(), m_gamepads.size());

    m_gamepads[gamepadData.index()] = makeUnique<WebGamepad>(gamepadData);
    m_rawGamepads[gamepadData.index()] = m_gamepads[gamepadData.index()].get();

    for (auto& client : m_clients)
        client.platformGamepadConnected(*m_gamepads[gamepadData.index()], eventVisibility);
}

void WebGamepadProvider::gamepadDisconnected(unsigned index)
{
    WP_MESSAGE_CHECK((m_gamepads.size() > index), index, m_gamepads.size());

    std::unique_ptr<WebGamepad> disconnectedGamepad = WTFMove(m_gamepads[index]);
    m_rawGamepads[index] = nullptr;

    LOG(Gamepad, "WebGamepadProvider::gamepadDisconnected - Gamepad index %u detached (m_gamepads size %zu, m_rawGamepads size %zu\n", index, m_gamepads.size(), m_rawGamepads.size());

    for (auto& client : m_clients)
        client.platformGamepadDisconnected(*disconnectedGamepad);
}

void WebGamepadProvider::gamepadActivity(const Vector<std::optional<GamepadData>>& gamepadDatas, EventMakesGamepadsVisible eventVisibility)
{
    LOG(Gamepad, "WebGamepadProvider::gamepadActivity - %zu gamepad datas with %zu local web gamepads\n", gamepadDatas.size(), m_gamepads.size());

    ASSERT(m_gamepads.size() == gamepadDatas.size());

    for (size_t i = 0; i < m_gamepads.size(); ++i) {
        if (m_gamepads[i] && gamepadDatas[i])
            m_gamepads[i]->updateValues(*gamepadDatas[i]);
    }

    for (auto& client : m_clients)
        client.platformGamepadInputActivity(eventVisibility);
}

void WebGamepadProvider::startMonitoringGamepads(GamepadProviderClient& client)
{
    bool processHadGamepadClients = !m_clients.isEmptyIgnoringNullReferences();

    ASSERT(!m_clients.contains(client));
    m_clients.add(client);

    if (!processHadGamepadClients)
        WebProcess::singleton().send(Messages::WebProcessPool::StartedUsingGamepads(), 0);
}

void WebGamepadProvider::stopMonitoringGamepads(GamepadProviderClient& client)
{
    ASSERT(m_clients.contains(client));
    m_clients.remove(client);
    if (!m_clients.isEmptyIgnoringNullReferences())
        return;

    WebProcess::singleton().sendWithAsyncReply(Messages::WebProcessPool::StoppedUsingGamepads(), [this] {
        m_gamepads.clear();
        m_rawGamepads.clear();
    });
}

const Vector<WeakPtr<PlatformGamepad>>& WebGamepadProvider::platformGamepads()
{
    return m_rawGamepads;
}

void WebGamepadProvider::playEffect(unsigned gamepadIndex, const String& gamepadID, GamepadHapticEffectType type, const GamepadEffectParameters& parameters, CompletionHandler<void(bool)>&& completionHandler)
{
    WebProcess::singleton().sendWithAsyncReply(Messages::WebProcessPool::PlayGamepadEffect(gamepadIndex, gamepadID, type, parameters), WTFMove(completionHandler));
}

void WebGamepadProvider::stopEffects(unsigned gamepadIndex, const String& gamepadID, CompletionHandler<void()>&& completionHandler)
{
    WebProcess::singleton().sendWithAsyncReply(Messages::WebProcessPool::StopGamepadEffects(gamepadIndex, gamepadID), WTFMove(completionHandler));
}

}

#endif // ENABLE(GAMEPAD)
