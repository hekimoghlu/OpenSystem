/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 26, 2023.
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

#include <wtf/CompletionHandler.h>
#include <wtf/Function.h>
#include <wtf/ProcessID.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/ThreadSafeWeakPtr.h>
#include <wtf/text/WTFString.h>

#if !OS(WINDOWS)
#include <unistd.h>
#endif

#if USE(EXTENSIONKIT)
#include "AssertionCapability.h"
#endif

#if USE(EXTENSIONKIT)
#include "ExtensionProcess.h"
#endif

#if USE(RUNNINGBOARD)
#include <wtf/RetainPtr.h>

OBJC_CLASS RBSAssertion;
OBJC_CLASS WKRBSAssertionDelegate;
#endif // USE(RUNNINGBOARD)

#if USE(EXTENSIONKIT)
OBJC_CLASS BEWebContentProcess;
OBJC_CLASS BENetworkingProcess;
OBJC_CLASS BERenderingProcess;
OBJC_PROTOCOL(BEProcessCapabilityGrant);
#endif

namespace WebKit {

enum class ProcessAssertionType : uint8_t {
    NearSuspended,
    Background,
    UnboundedNetworking,
    Foreground,
    MediaPlayback,
    FinishTaskCanSleep,
    FinishTaskInterruptable,
    BoostedJetsam,
};

ASCIILiteral processAssertionTypeDescription(ProcessAssertionType);

class AuxiliaryProcessProxy;

class ProcessAssertion : public ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr<ProcessAssertion> {
    WTF_MAKE_TZONE_ALLOCATED(ProcessAssertion);
public:
    enum class Mode : bool { Sync, Async };
    static Ref<ProcessAssertion> create(ProcessID, const String& reason, ProcessAssertionType, Mode = Mode::Async, const String& environmentIdentifier = emptyString(), CompletionHandler<void()>&& acquisisionHandler = nullptr);
    static Ref<ProcessAssertion> create(AuxiliaryProcessProxy&, const String& reason, ProcessAssertionType, Mode = Mode::Async, CompletionHandler<void()>&& acquisisionHandler = nullptr);

    static double remainingRunTimeInSeconds(ProcessID);
    virtual ~ProcessAssertion();

    void setPrepareForInvalidationHandler(Function<void()>&& handler) { m_prepareForInvalidationHandler = WTFMove(handler); }
    void setInvalidationHandler(Function<void()>&& handler) { m_invalidationHandler = WTFMove(handler); }

    ProcessAssertionType type() const { return m_assertionType; }
    ProcessID pid() const { return m_pid; }

    bool isValid() const;

protected:
#if !USE(EXTENSIONKIT)
    ProcessAssertion(ProcessID, const String& reason, ProcessAssertionType, const String& environmentIdentifier);
#else
    ProcessAssertion(ProcessID, const String& reason, ProcessAssertionType, const String& environmentIdentifier, std::optional<ExtensionProcess>&&);
#endif

    void init(const String& environmentIdentifier);

    void acquireAssertion(Mode, CompletionHandler<void()>&&);

    void acquireAsync(CompletionHandler<void()>&&);
    void acquireSync();

#if USE(RUNNINGBOARD)
    void processAssertionWillBeInvalidated();
    virtual void processAssertionWasInvalidated();
#endif

private:
    const ProcessAssertionType m_assertionType;
    const ProcessID m_pid;
    const String m_reason;
#if USE(RUNNINGBOARD)
    RetainPtr<RBSAssertion> m_rbsAssertion;
    RetainPtr<WKRBSAssertionDelegate> m_delegate;
    bool m_wasInvalidated { false };
#endif
    Function<void()> m_prepareForInvalidationHandler;
    Function<void()> m_invalidationHandler;
#if USE(EXTENSIONKIT)
    static Lock s_capabilityLock;
    std::optional<AssertionCapability> m_capability;
    ExtensionCapabilityGrant m_grant WTF_GUARDED_BY_LOCK(s_capabilityLock);
    std::optional<ExtensionProcess> m_process;
#endif
};

class ProcessAndUIAssertion final : public ProcessAssertion {
    WTF_MAKE_TZONE_ALLOCATED(ProcessAndUIAssertion);
public:
    static Ref<ProcessAndUIAssertion> create(AuxiliaryProcessProxy&, const String& reason, ProcessAssertionType, Mode = Mode::Async, CompletionHandler<void()>&& acquisisionHandler = nullptr);
#if !USE(EXTENSIONKIT)
    static Ref<ProcessAndUIAssertion> create(ProcessID, const String& reason, ProcessAssertionType, const String& environmentIdentifier, Mode = Mode::Async, CompletionHandler<void()>&& acquisisionHandler = nullptr);
#else
    static Ref<ProcessAndUIAssertion> create(ProcessID, const String& reason, ProcessAssertionType, const String& environmentIdentifier, std::optional<ExtensionProcess>&&, Mode = Mode::Async, CompletionHandler<void()>&& acquisisionHandler = nullptr);
#endif

    ~ProcessAndUIAssertion();

    void uiAssertionWillExpireImminently();

    void setUIAssertionExpirationHandler(Function<void()>&& handler) { m_uiAssertionExpirationHandler = WTFMove(handler); }
#if PLATFORM(IOS_FAMILY)
    static void setProcessStateMonitorEnabled(bool);
#endif

private:
#if !USE(EXTENSIONKIT)
    ProcessAndUIAssertion(ProcessID, const String& reason, ProcessAssertionType, const String& environmentIdentifier);
#else
    ProcessAndUIAssertion(ProcessID, const String& reason, ProcessAssertionType, const String& environmentIdentifier, std::optional<ExtensionProcess>&&);
#endif

#if PLATFORM(IOS_FAMILY)
    void processAssertionWasInvalidated() final;
#endif
    void updateRunInBackgroundCount();

    Function<void()> m_uiAssertionExpirationHandler;
    bool m_isHoldingBackgroundTask { false };
};
    
} // namespace WebKit
