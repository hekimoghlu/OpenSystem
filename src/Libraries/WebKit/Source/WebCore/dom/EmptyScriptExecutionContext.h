/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 13, 2022.
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

#include "AdvancedPrivacyProtections.h"
#include "EventLoop.h"
#include "Microtasks.h"
#include "ReferrerPolicy.h"
#include "ScriptExecutionContext.h"
#include "SecurityOrigin.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class EmptyScriptExecutionContext final : public RefCounted<EmptyScriptExecutionContext>, public ScriptExecutionContext {
public:
    static Ref<EmptyScriptExecutionContext> create(JSC::VM& vm)
    {
        return adoptRef(*new EmptyScriptExecutionContext(vm));
    }

    ~EmptyScriptExecutionContext()
    {
        m_eventLoop->removeAssociatedContext(*this);
    }

    bool isSecureContext() const final { return false; }
    bool isJSExecutionForbidden() const final { return false; }
    EventLoopTaskGroup& eventLoop() final
    {
        ASSERT_NOT_REACHED();
        return *m_eventLoopTaskGroup;
    }
    const URL& url() const final { return m_url; }
    const URL& cookieURL() const final { return url(); }
    URL completeURL(const String&, ForceUTF8 = ForceUTF8::No) const final { return { }; };
    String userAgent(const URL&) const final { return emptyString(); }
    ReferrerPolicy referrerPolicy() const final { return ReferrerPolicy::EmptyString; }

    void disableEval(const String&) final { };
    void disableWebAssembly(const String&) final { };
    void setRequiresTrustedTypes(bool) final { };

    IDBClient::IDBConnectionProxy* idbConnectionProxy() final { return nullptr; }
    SocketProvider* socketProvider() final { return nullptr; }

    void addConsoleMessage(std::unique_ptr<Inspector::ConsoleMessage>&&) final { }
    void addConsoleMessage(MessageSource, MessageLevel, const String&, unsigned long) final { };

    SecurityOrigin& topOrigin() const final { return m_origin.get(); };

    OptionSet<AdvancedPrivacyProtections> advancedPrivacyProtections() const final { return { }; }
    std::optional<uint64_t> noiseInjectionHashSalt() const { return std::nullopt; }
    OptionSet<NoiseInjectionPolicy> noiseInjectionPolicies() const { return { }; }

    void postTask(Task&&) final { ASSERT_NOT_REACHED(); }
    EventTarget* errorEventTarget() final { return nullptr; };

    std::optional<Vector<uint8_t>> wrapCryptoKey(const Vector<uint8_t>&) final { return std::nullopt; }
    std::optional<Vector<uint8_t>> serializeAndWrapCryptoKey(CryptoKeyData&&) final { return std::nullopt; }
    std::optional<Vector<uint8_t>> unwrapCryptoKey(const Vector<uint8_t>&) final { return std::nullopt; }

    JSC::VM& vm() final { return m_vm; }
    JSC::VM* vmIfExists() const final { return m_vm.ptr(); }

    using RefCounted::ref;
    using RefCounted::deref;

private:
    EmptyScriptExecutionContext(JSC::VM& vm)
        : ScriptExecutionContext(Type::EmptyScriptExecutionContext)
        , m_vm(vm)
        , m_origin(SecurityOrigin::createOpaque())
        , m_eventLoop(EmptyEventLoop::create(vm))
        , m_eventLoopTaskGroup(makeUnique<EventLoopTaskGroup>(m_eventLoop))
    {
        relaxAdoptionRequirement();
        m_eventLoop->addAssociatedContext(*this);
    }

    void addMessage(MessageSource, MessageLevel, const String&, const String&, unsigned, unsigned, RefPtr<Inspector::ScriptCallStack>&&, JSC::JSGlobalObject* = nullptr, unsigned long = 0) final { }
    void logExceptionToConsole(const String&, const String&, int, int, RefPtr<Inspector::ScriptCallStack>&&) final { }

    const Settings::Values& settingsValues() const final { return m_settingsValues; }

#if ENABLE(NOTIFICATIONS)
    NotificationClient* notificationClient() final { return nullptr; }
#endif

    class EmptyEventLoop final : public EventLoop {
    public:
        static Ref<EmptyEventLoop> create(JSC::VM& vm)
        {
            return adoptRef(*new EmptyEventLoop(vm));
        }

        MicrotaskQueue& microtaskQueue() final { return m_queue; };

    private:
        explicit EmptyEventLoop(JSC::VM& vm)
            : m_queue(MicrotaskQueue(vm, *this))
        {
        }

        void scheduleToRun() final { ASSERT_NOT_REACHED(); }
        bool isContextThread() const final { return true; }

        MicrotaskQueue m_queue;
    };

    Ref<JSC::VM> m_vm;
    Ref<SecurityOrigin> m_origin;
    URL m_url;
    Ref<EmptyEventLoop> m_eventLoop;
    std::unique_ptr<EventLoopTaskGroup> m_eventLoopTaskGroup;
    Settings::Values m_settingsValues;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::EmptyScriptExecutionContext)
    static bool isType(const WebCore::ScriptExecutionContext& context) { return context.isEmptyScriptExecutionContext(); }
SPECIALIZE_TYPE_TRAITS_END()
