/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 20, 2021.
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

#if ENABLE(REMOTE_INSPECTOR)

#include "InspectorFrontendChannel.h"
#include "RemoteControllableTarget.h"
#include <wtf/Lock.h>
#include <wtf/ThreadSafeRefCounted.h>
#include <wtf/ThreadSafeWeakPtr.h>

#if PLATFORM(COCOA)
#include <wtf/Function.h>
#include <wtf/RetainPtr.h>

OBJC_CLASS NSString;
#endif

namespace Inspector {

class RemoteControllableTarget;

#if PLATFORM(COCOA)
typedef Vector<Function<void ()>> RemoteTargetQueue;
#endif

class RemoteConnectionToTarget final : public ThreadSafeRefCounted<RemoteConnectionToTarget>, public FrontendChannel {
public:
#if PLATFORM(COCOA)
    RemoteConnectionToTarget(RemoteControllableTarget*, NSString* connectionIdentifier, NSString* destination);
#else
    RemoteConnectionToTarget(RemoteControllableTarget&);
#endif
    ~RemoteConnectionToTarget() final;

    // Main API.
    bool setup(bool isAutomaticInspection = false, bool automaticallyPause = false);
#if PLATFORM(COCOA)
    void sendMessageToTarget(NSString *);
#else
    void sendMessageToTarget(String&&);
#endif
    void close();
    void targetClosed();

    std::optional<TargetID> targetIdentifier() const WTF_REQUIRES_LOCK(m_targetMutex);
#if PLATFORM(COCOA)
    NSString *connectionIdentifier() const;
    NSString *destination() const;

    Lock& queueMutex() { return m_queueMutex; }
    const RemoteTargetQueue& queue() const { return m_queue; }
    RemoteTargetQueue takeQueue();
#endif

    // FrontendChannel overrides.
    ConnectionType connectionType() const final { return ConnectionType::Remote; }
    void sendMessageToFrontend(const String&) final;

private:
#if PLATFORM(COCOA)
    void dispatchAsyncOnTarget(Function<void ()>&&);

    void setupRunLoop();
    void teardownRunLoop();
    void queueTaskOnPrivateRunLoop(Function<void ()>&&);
#endif

    // This connection from the RemoteInspector singleton to the InspectionTarget
    // can be used on multiple threads. So any access to the target
    // itself must take this mutex to ensure m_target is valid.
    mutable Lock m_targetMutex;

#if PLATFORM(COCOA)
    // If a target has a specific run loop it wants to evaluate on
    // we setup our run loop sources on that specific run loop.
    RetainPtr<CFRunLoopRef> m_runLoop;
    RetainPtr<CFRunLoopSourceRef> m_runLoopSource;
    RemoteTargetQueue m_queue;
    Lock m_queueMutex;
#endif

    ThreadSafeWeakPtr<RemoteControllableTarget> m_target WTF_GUARDED_BY_LOCK(m_targetMutex);
    bool m_connected { false };

#if PLATFORM(COCOA)
    RetainPtr<NSString> m_connectionIdentifier;
    RetainPtr<NSString> m_destination;
#endif
};

} // namespace Inspector

#endif // ENABLE(REMOTE_INSPECTOR)
