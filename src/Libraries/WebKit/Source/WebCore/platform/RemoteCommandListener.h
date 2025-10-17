/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 21, 2023.
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
#pragma once

#include "DeferrableTask.h"
#include "PlatformMediaSession.h"
#include <wtf/AbstractRefCounted.h>

namespace WebCore {

class RemoteCommandListenerClient {
public:
    virtual ~RemoteCommandListenerClient() = default;
    virtual void didReceiveRemoteControlCommand(PlatformMediaSession::RemoteControlCommandType, const PlatformMediaSession::RemoteCommandArgument&) = 0;
};

class WEBCORE_EXPORT RemoteCommandListener : public AbstractRefCounted {
public:
    static RefPtr<RemoteCommandListener> create(RemoteCommandListenerClient&);
    RemoteCommandListener(RemoteCommandListenerClient&);
    virtual ~RemoteCommandListener();

    using CreationFunction = Function<RefPtr<RemoteCommandListener>(RemoteCommandListenerClient&)>;
    static void setCreationFunction(CreationFunction&&);
    static void resetCreationFunction();

    void addSupportedCommand(PlatformMediaSession::RemoteControlCommandType);
    void removeSupportedCommand(PlatformMediaSession::RemoteControlCommandType);

    using RemoteCommandsSet = UncheckedKeyHashSet<PlatformMediaSession::RemoteControlCommandType, IntHash<PlatformMediaSession::RemoteControlCommandType>, WTF::StrongEnumHashTraits<PlatformMediaSession::RemoteControlCommandType>>;
    void setSupportedCommands(const RemoteCommandsSet&);
    const RemoteCommandsSet& supportedCommands() const;

    virtual void updateSupportedCommands();
    void scheduleSupportedCommandsUpdate();

    void setSupportsSeeking(bool);
    bool supportsSeeking() const;

    RemoteCommandListenerClient& client() const { return m_client; }

private:
    RemoteCommandListenerClient& m_client;
    RemoteCommandsSet m_supportedCommands;
    MainThreadDeferrableTask m_updateCommandsTask;
    bool m_supportsSeeking { false };
};

}
