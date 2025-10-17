/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 3, 2022.
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

#include "ExceptionOr.h"
#include "ScriptWrappable.h"
#include "SubscriberCallback.h"
#include "VoidCallback.h"
#include <wtf/RefCounted.h>

namespace WebCore {

class DeferredPromise;
class InternalObserver;
class JSSubscriptionObserverCallback;
class MapperCallback;
class PredicateCallback;
class ReducerCallback;
class ScriptExecutionContext;
class VisitorCallback;
struct ObservableInspector;
struct SubscribeOptions;
struct SubscriptionObserver;

class Observable final : public ScriptWrappable, public RefCounted<Observable> {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(Observable);

public:
    using ObserverUnion = std::variant<RefPtr<JSSubscriptionObserverCallback>, SubscriptionObserver>;
    using InspectorUnion = std::variant<RefPtr<JSSubscriptionObserverCallback>, ObservableInspector>;

    static Ref<Observable> create(Ref<SubscriberCallback>);

    explicit Observable(Ref<SubscriberCallback>);

    void subscribe(ScriptExecutionContext&, std::optional<ObserverUnion>, SubscribeOptions);
    void subscribeInternal(ScriptExecutionContext&, Ref<InternalObserver>&&, const SubscribeOptions&);

    Ref<Observable> map(ScriptExecutionContext&, MapperCallback&);
    Ref<Observable> filter(ScriptExecutionContext&, PredicateCallback&);
    Ref<Observable> take(ScriptExecutionContext&, uint64_t);
    Ref<Observable> drop(ScriptExecutionContext&, uint64_t);
    Ref<Observable> inspect(ScriptExecutionContext&, std::optional<InspectorUnion>&&);

    // Promise-returning operators.

    void first(ScriptExecutionContext&, const SubscribeOptions&, Ref<DeferredPromise>&&);
    void forEach(ScriptExecutionContext&, Ref<VisitorCallback>&&, const SubscribeOptions&, Ref<DeferredPromise>&&);
    void last(ScriptExecutionContext&, const SubscribeOptions&, Ref<DeferredPromise>&&);
    void find(ScriptExecutionContext&, Ref<PredicateCallback>&&, const SubscribeOptions&, Ref<DeferredPromise>&&);
    void every(ScriptExecutionContext&, Ref<PredicateCallback>&&, const SubscribeOptions&, Ref<DeferredPromise>&&);
    void some(ScriptExecutionContext&, Ref<PredicateCallback>&&, const SubscribeOptions&, Ref<DeferredPromise>&&);
    void reduce(ScriptExecutionContext&, Ref<ReducerCallback>&&, JSC::JSValue, const SubscribeOptions&, Ref<DeferredPromise>&&);

private:
    Ref<SubscriberCallback> m_subscriberCallback;
};

} // namespace WebCore
