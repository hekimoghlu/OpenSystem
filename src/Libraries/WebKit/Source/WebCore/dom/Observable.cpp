/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 22, 2025.
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
#include "Observable.h"

#include "AbortSignal.h"
#include "CallbackResult.h"
#include "Document.h"
#include "Exception.h"
#include "ExceptionCode.h"
#include "InternalObserverDrop.h"
#include "InternalObserverEvery.h"
#include "InternalObserverFilter.h"
#include "InternalObserverFind.h"
#include "InternalObserverFirst.h"
#include "InternalObserverForEach.h"
#include "InternalObserverFromScript.h"
#include "InternalObserverInspect.h"
#include "InternalObserverLast.h"
#include "InternalObserverMap.h"
#include "InternalObserverReduce.h"
#include "InternalObserverSome.h"
#include "InternalObserverTake.h"
#include "JSDOMPromiseDeferred.h"
#include "JSSubscriptionObserverCallback.h"
#include "MapperCallback.h"
#include "ObservableInspector.h"
#include "PredicateCallback.h"
#include "ReducerCallback.h"
#include "SubscribeOptions.h"
#include "Subscriber.h"
#include "SubscriberCallback.h"
#include "SubscriptionObserver.h"
#include "VisitorCallback.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(Observable);

Ref<Observable> Observable::create(Ref<SubscriberCallback> callback)
{
    return adoptRef(*new Observable(callback));
}

void Observable::subscribe(ScriptExecutionContext& context, std::optional<ObserverUnion> observer, SubscribeOptions options)
{
    if (observer) {
        WTF::switchOn(
            observer.value(),
            [&](RefPtr<JSSubscriptionObserverCallback>& next) {
                subscribeInternal(context, InternalObserverFromScript::create(context, next), options);
            },
            [&](SubscriptionObserver& subscription) {
                subscribeInternal(context, InternalObserverFromScript::create(context, subscription), options);
            }
        );
    } else
        subscribeInternal(context, InternalObserverFromScript::create(context, nullptr), options);
}

void Observable::subscribeInternal(ScriptExecutionContext& context, Ref<InternalObserver>&& observer, const SubscribeOptions& options)
{
    RefPtr document = dynamicDowncast<Document>(context);
    if (document && !document->isFullyActive())
        return;

    Ref subscriber = Subscriber::create(context, WTFMove(observer), options);

    Ref vm = context.globalObject()->vm();
    JSC::JSLockHolder lock(vm);

    // The exception is not reported, instead it is forwarded to the
    // error handler.
    JSC::Exception* previousException = nullptr;
    {
        auto catchScope = DECLARE_CATCH_SCOPE(vm);
        m_subscriberCallback->handleEventRethrowingException(subscriber);
        previousException = catchScope.exception();
        if (previousException) {
            catchScope.clearException();
            subscriber->error(previousException->value());
        }
    }
}

Ref<Observable> Observable::map(ScriptExecutionContext& context, MapperCallback& mapper)
{
    return create(createSubscriberCallbackMap(context, *this, mapper));
}

Ref<Observable> Observable::filter(ScriptExecutionContext& context, PredicateCallback& predicate)
{
    return create(createSubscriberCallbackFilter(context, *this, predicate));
}

Ref<Observable> Observable::take(ScriptExecutionContext& context, uint64_t amount)
{
    return create(createSubscriberCallbackTake(context, *this, amount));
}

Ref<Observable> Observable::drop(ScriptExecutionContext& context, uint64_t amount)
{
    return create(createSubscriberCallbackDrop(context, *this, amount));
}

Ref<Observable> Observable::inspect(ScriptExecutionContext& context, std::optional<InspectorUnion>&& inspectorUnion)
{
    if (!inspectorUnion)
        return *this;

    return WTF::switchOn(WTFMove(*inspectorUnion),
        [&](RefPtr<JSSubscriptionObserverCallback>&& next) {
            return create(createSubscriberCallbackInspect(context, *this, WTFMove(next)));
        },
        [&](ObservableInspector&& inspector) {
            return create(createSubscriberCallbackInspect(context, *this, WTFMove(inspector)));
        }
    );
}

void Observable::first(ScriptExecutionContext& context, const SubscribeOptions& options, Ref<DeferredPromise>&& promise)
{
    return createInternalObserverOperatorFirst(context, *this, options, WTFMove(promise));
}

void Observable::forEach(ScriptExecutionContext& context, Ref<VisitorCallback>&& callback, const SubscribeOptions& options, Ref<DeferredPromise>&& promise)
{
    return createInternalObserverOperatorForEach(context, *this, WTFMove(callback), options, WTFMove(promise));
}

void Observable::last(ScriptExecutionContext& context, const SubscribeOptions& options, Ref<DeferredPromise>&& promise)
{
    return createInternalObserverOperatorLast(context, *this, options, WTFMove(promise));
}

void Observable::find(ScriptExecutionContext& context, Ref<PredicateCallback>&& callback, const SubscribeOptions& options, Ref<DeferredPromise>&& promise)
{
    return createInternalObserverOperatorFind(context, *this, WTFMove(callback), options, WTFMove(promise));
}

void Observable::every(ScriptExecutionContext& context, Ref<PredicateCallback>&& callback, const SubscribeOptions& options, Ref<DeferredPromise>&& promise)
{
    return createInternalObserverOperatorEvery(context, *this, WTFMove(callback), options, WTFMove(promise));
}

void Observable::some(ScriptExecutionContext& context, Ref<PredicateCallback>&& callback, const SubscribeOptions& options, Ref<DeferredPromise>&& promise)
{
    return createInternalObserverOperatorSome(context, *this, WTFMove(callback), options, WTFMove(promise));
}

void Observable::reduce(ScriptExecutionContext& context, Ref<ReducerCallback>&& callback, JSC::JSValue initialValue, const SubscribeOptions& options, Ref<DeferredPromise>&& promise)
{
    return createInternalObserverOperatorReduce(context, *this, WTFMove(callback), initialValue, options, WTFMove(promise));
}

Observable::Observable(Ref<SubscriberCallback> callback)
    : m_subscriberCallback(callback)
{
}

} // namespace WebCore
