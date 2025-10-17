/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 12, 2024.
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
#import "config.h"
#import "RemoteConnectionToTarget.h"

#if ENABLE(REMOTE_INSPECTOR)

#import "JSGlobalObjectDebugger.h"
#import "RemoteAutomationTarget.h"
#import "RemoteInspectionTarget.h"
#import "RemoteInspector.h"
#import <dispatch/dispatch.h>
#import <wtf/RunLoop.h>

#if USE(WEB_THREAD)
#import <wtf/ios/WebCoreThread.h>
#endif

namespace Inspector {

static Lock rwiQueueMutex;
static CFRunLoopSourceRef rwiRunLoopSource;
static RemoteTargetQueue* rwiQueue;

static void RemoteTargetHandleRunSourceGlobal(void*)
{
    ASSERT(CFRunLoopGetCurrent() == CFRunLoopGetMain());
    ASSERT(rwiRunLoopSource);
    ASSERT(rwiQueue);

    RemoteTargetQueue queueCopy;
    {
        Locker locker { rwiQueueMutex };
        std::swap(queueCopy, *rwiQueue);
    }

    for (const auto& function : queueCopy)
        function();
}

static void RemoteTargetQueueTaskOnGlobalQueue(Function<void ()>&& function)
{
    ASSERT(rwiRunLoopSource);
    ASSERT(rwiQueue);

    {
        Locker locker { rwiQueueMutex };
        rwiQueue->append(WTFMove(function));
    }

    CFRunLoopSourceSignal(rwiRunLoopSource);
    CFRunLoopWakeUp(CFRunLoopGetMain());
}

static void RemoteTargetInitializeGlobalQueue()
{
    static dispatch_once_t pred;
    dispatch_once(&pred, ^{
        rwiQueue = new RemoteTargetQueue;

        CFRunLoopSourceContext runLoopSourceContext = { 0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, RemoteTargetHandleRunSourceGlobal };
        rwiRunLoopSource = CFRunLoopSourceCreate(kCFAllocatorDefault, 1, &runLoopSourceContext);

        // Add to the default run loop mode for default handling, and the JSContext remote inspector run loop mode when paused.
        CFRunLoopAddSource(CFRunLoopGetMain(), rwiRunLoopSource, kCFRunLoopDefaultMode);
        auto mode = JSGlobalObjectDebugger::runLoopMode();
        if (mode != DefaultRunLoopMode)
            CFRunLoopAddSource(CFRunLoopGetMain(), rwiRunLoopSource, mode);
    });
}

static void RemoteTargetHandleRunSourceWithInfo(void* info)
{
    RemoteConnectionToTarget *connectionToTarget = static_cast<RemoteConnectionToTarget*>(info);

    RemoteTargetQueue queueCopy;
    {
        Locker locker { connectionToTarget->queueMutex() };
        queueCopy = connectionToTarget->takeQueue();
    }

    for (const auto& function : queueCopy)
        function();
}


RemoteConnectionToTarget::RemoteConnectionToTarget(RemoteControllableTarget* target, NSString *connectionIdentifier, NSString *destination)
    : m_target(target)
    , m_connectionIdentifier(connectionIdentifier)
    , m_destination(destination)
{
    setupRunLoop();
}

RemoteConnectionToTarget::~RemoteConnectionToTarget()
{
    teardownRunLoop();
}

std::optional<TargetID> RemoteConnectionToTarget::targetIdentifier() const
{
    RefPtr target = m_target.get();
    return target ? std::optional<TargetID>(target->targetIdentifier()) : std::nullopt;
}

NSString *RemoteConnectionToTarget::connectionIdentifier() const
{
    return adoptNS([m_connectionIdentifier copy]).autorelease();
}

NSString *RemoteConnectionToTarget::destination() const
{
    return adoptNS([m_destination copy]).autorelease();
}

void RemoteConnectionToTarget::dispatchAsyncOnTarget(Function<void ()>&& callback)
{
    if (m_runLoop) {
        queueTaskOnPrivateRunLoop(WTFMove(callback));
        return;
    }

#if USE(WEB_THREAD)
    if (WebCoreWebThreadIsEnabled && WebCoreWebThreadIsEnabled()) {
        __block auto blockCallback(WTFMove(callback));
        WebCoreWebThreadRun(^{
            blockCallback();
        });
        return;
    }
#endif

    RemoteTargetQueueTaskOnGlobalQueue(WTFMove(callback));
}

bool RemoteConnectionToTarget::setup(bool isAutomaticInspection, bool automaticallyPause)
{
    Locker locker { m_targetMutex };

    RefPtr target = m_target.get();
    if (!target)
        return false;

    auto targetIdentifier = this->targetIdentifier().value_or(0);

    dispatchAsyncOnTarget([this, targetIdentifier, isAutomaticInspection, automaticallyPause, protectedThis = Ref { *this }]() {
        RefPtr<RemoteControllableTarget> target;
        {
            Locker locker { m_targetMutex };
            target = m_target.get();
        }
        if (!target || !target->remoteControlAllowed()) {
            RemoteInspector::singleton().setupFailed(targetIdentifier);
            Locker locker { m_targetMutex };
            m_target = nullptr;
        } else if (auto* inspectionTarget = dynamicDowncast<RemoteInspectionTarget>(target.get())) {
            inspectionTarget->connect(*this, isAutomaticInspection, automaticallyPause);
            m_connected = true;

            RemoteInspector::singleton().updateTargetListing(targetIdentifier);
        } else if (auto* automationTarget = dynamicDowncast<RemoteAutomationTarget>(target.get())) {
            automationTarget->connect(*this);
            m_connected = true;

            RemoteInspector::singleton().updateTargetListing(targetIdentifier);
        }
    });

    return true;
}

void RemoteConnectionToTarget::targetClosed()
{
    Locker locker { m_targetMutex };

    m_target = nullptr;
}

void RemoteConnectionToTarget::close()
{
    Locker locker { m_targetMutex };
    RefPtr target = m_target.get();
    auto targetIdentifier = target ? target->targetIdentifier() : 0;

    if (auto* automationTarget = dynamicDowncast<RemoteAutomationTarget>(target.get()))
        automationTarget->setIsPendingTermination();
    
    dispatchAsyncOnTarget([this, targetIdentifier, protectedThis = Ref { *this }]() {
        Locker locker { m_targetMutex };
        if (RefPtr target = m_target.get()) {
            if (m_connected)
                target->disconnect(*this);

            m_target = nullptr;
        }

        if (targetIdentifier)
            RemoteInspector::singleton().updateTargetListing(targetIdentifier);
    });
}

void RemoteConnectionToTarget::sendMessageToTarget(NSString *message)
{
    dispatchAsyncOnTarget([this, strongMessage = retainPtr(message), protectedThis = Ref { *this }]() {
        RefPtr<RemoteControllableTarget> target;
        {
            Locker locker { m_targetMutex };
            target = m_target.get();
        }
        if (target)
            target->dispatchMessageFromRemote(strongMessage.get());
    });
}

void RemoteConnectionToTarget::sendMessageToFrontend(const String& message)
{
    std::optional<TargetID> targetIdentifier;
    {
        Locker locker { m_targetMutex };
        targetIdentifier = this->targetIdentifier();
    }
    if (!targetIdentifier)
        return;

    RemoteInspector::singleton().sendMessageToRemote(targetIdentifier.value(), message);
}

void RemoteConnectionToTarget::setupRunLoop()
{
    RetainPtr<CFRunLoopRef> targetRunLoop;
    {
        Locker locker { m_targetMutex };
        RefPtr target = m_target.get();
        targetRunLoop = target->targetRunLoop();
    }
    if (!targetRunLoop) {
        RemoteTargetInitializeGlobalQueue();
        return;
    }

    m_runLoop = targetRunLoop;

    CFRunLoopSourceContext runLoopSourceContext = { 0, this, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, RemoteTargetHandleRunSourceWithInfo };
    m_runLoopSource = adoptCF(CFRunLoopSourceCreate(kCFAllocatorDefault, 1, &runLoopSourceContext));

    CFRunLoopAddSource(m_runLoop.get(), m_runLoopSource.get(), kCFRunLoopDefaultMode);
    auto mode = JSGlobalObjectDebugger::runLoopMode();
    if (mode != DefaultRunLoopMode)
        CFRunLoopAddSource(m_runLoop.get(), m_runLoopSource.get(), mode);
}

void RemoteConnectionToTarget::teardownRunLoop()
{
    if (!m_runLoop)
        return;

    CFRunLoopRemoveSource(m_runLoop.get(), m_runLoopSource.get(), kCFRunLoopDefaultMode);
    auto mode = JSGlobalObjectDebugger::runLoopMode();
    if (mode != DefaultRunLoopMode)
        CFRunLoopRemoveSource(m_runLoop.get(), m_runLoopSource.get(), mode);

    m_runLoop = nullptr;
    m_runLoopSource = nullptr;
}

void RemoteConnectionToTarget::queueTaskOnPrivateRunLoop(Function<void ()>&& function)
{
    ASSERT(m_runLoop);

    {
        Locker lock { m_queueMutex };
        m_queue.append(WTFMove(function));
    }

    CFRunLoopSourceSignal(m_runLoopSource.get());
    CFRunLoopWakeUp(m_runLoop.get());
}

RemoteTargetQueue RemoteConnectionToTarget::takeQueue()
{
    return std::exchange(m_queue, { });
}

} // namespace Inspector

#endif // ENABLE(REMOTE_INSPECTOR)
