/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 4, 2024.
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

#include "ActiveDOMObject.h"
#include "CookieChangeListener.h"
#include "CookieJar.h"
#include "EventTarget.h"
#include <wtf/Forward.h>
#include <wtf/RefCounted.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

struct CookieInit;
struct CookieStoreDeleteOptions;
struct CookieStoreGetOptions;
class Document;
class DeferredPromise;
class ScriptExecutionContext;

class CookieStore final : public RefCounted<CookieStore>, public EventTarget, public ActiveDOMObject, public CookieChangeListener {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(CookieStore);
public:
    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    USING_CAN_MAKE_WEAKPTR(EventTarget);

    static Ref<CookieStore> create(ScriptExecutionContext*);
    ~CookieStore();

    void get(String&& name, Ref<DeferredPromise>&&);
    void get(CookieStoreGetOptions&&, Ref<DeferredPromise>&&);

    void getAll(String&& name, Ref<DeferredPromise>&&);
    void getAll(CookieStoreGetOptions&&, Ref<DeferredPromise>&&);

    void set(String&& name, String&& value, Ref<DeferredPromise>&&);
    void set(CookieInit&&, Ref<DeferredPromise>&&);

    void remove(String&& name, Ref<DeferredPromise>&&);
    void remove(CookieStoreDeleteOptions&&, Ref<DeferredPromise>&&);

private:
    explicit CookieStore(ScriptExecutionContext*);

    // CookieChangeListener
    void cookiesAdded(const String& host, const Vector<Cookie>&) final;
    void cookiesDeleted(const String& host, const Vector<Cookie>&) final;

    // ActiveDOMObject
    void stop() final;
    bool virtualHasPendingActivity() const final;

    // EventTarget
    enum EventTargetInterfaceType eventTargetInterface() const final;
    ScriptExecutionContext* scriptExecutionContext() const final;
    void refEventTarget() final { ref(); }
    void derefEventTarget() final { deref(); }
    void eventListenersDidChange() final;

    RefPtr<DeferredPromise> takePromise(uint64_t promiseIdentifier);

    class MainThreadBridge;
    Ref<MainThreadBridge> m_mainThreadBridge;
    Ref<MainThreadBridge> protectedMainThreadBridge() const;

    bool m_hasChangeEventListener { false };
    WeakPtr<CookieJar> m_cookieJar;
    String m_host;
    uint64_t m_nextPromiseIdentifier { 0 };
    HashMap<uint64_t, Ref<DeferredPromise>> m_promises;
};

}
