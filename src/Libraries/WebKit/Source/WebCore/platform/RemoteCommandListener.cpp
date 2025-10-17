/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 7, 2023.
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
#include "RemoteCommandListener.h"

#if PLATFORM(COCOA)
#include "RemoteCommandListenerCocoa.h"
#endif

#if USE(GLIB)
#include "RemoteCommandListenerGLib.h"
#endif

#include <wtf/NeverDestroyed.h>

namespace WebCore {

static RemoteCommandListener::CreationFunction& remoteCommandListenerCreationFunction()
{
    static NeverDestroyed<RemoteCommandListener::CreationFunction> creationFunction;
    return creationFunction;
}

void RemoteCommandListener::setCreationFunction(CreationFunction&& function)
{
    remoteCommandListenerCreationFunction() = WTFMove(function);
}

void RemoteCommandListener::resetCreationFunction()
{
    remoteCommandListenerCreationFunction() = [] (RemoteCommandListenerClient& client) -> RefPtr<RemoteCommandListener> {
#if PLATFORM(COCOA)
        return RemoteCommandListenerCocoa::create(client);
#elif USE(GLIB) && ENABLE(MEDIA_SESSION)
        return RemoteCommandListenerGLib::create(client);
#else
        UNUSED_PARAM(client);
        return nullptr;
#endif
    };
}

RefPtr<RemoteCommandListener> RemoteCommandListener::create(RemoteCommandListenerClient& client)
{
    if (!remoteCommandListenerCreationFunction())
        resetCreationFunction();
    return remoteCommandListenerCreationFunction()(client);
}

RemoteCommandListener::RemoteCommandListener(RemoteCommandListenerClient& client)
    : m_client(client)
{
}

RemoteCommandListener::~RemoteCommandListener() = default;


void RemoteCommandListener::scheduleSupportedCommandsUpdate()
{
    if (!m_updateCommandsTask.isPending()) {
        m_updateCommandsTask.scheduleTask([this] ()  {
            updateSupportedCommands();
        });
    }
}

void RemoteCommandListener::setSupportsSeeking(bool supports)
{
    if (m_supportsSeeking == supports)
        return;

    m_supportsSeeking = supports;
    scheduleSupportedCommandsUpdate();
}

void RemoteCommandListener::addSupportedCommand(PlatformMediaSession::RemoteControlCommandType command)
{
    m_supportedCommands.add(command);
    scheduleSupportedCommandsUpdate();
}

void RemoteCommandListener::removeSupportedCommand(PlatformMediaSession::RemoteControlCommandType command)
{
    m_supportedCommands.remove(command);
    scheduleSupportedCommandsUpdate();
}

void RemoteCommandListener::setSupportedCommands(const RemoteCommandsSet& commands)
{
    m_supportedCommands = commands;
    scheduleSupportedCommandsUpdate();
}

void RemoteCommandListener::updateSupportedCommands()
{
    ASSERT_NOT_REACHED();
}

const RemoteCommandListener::RemoteCommandsSet& RemoteCommandListener::supportedCommands() const
{
    return m_supportedCommands;
}

bool RemoteCommandListener::supportsSeeking() const
{
    return m_supportsSeeking;
}

}
