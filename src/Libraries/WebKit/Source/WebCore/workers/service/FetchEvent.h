/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 2, 2022.
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

#include "ExtendableEvent.h"
#include "FetchIdentifier.h"
#include "JSDOMPromiseDeferredForward.h"
#include "ResourceError.h"
#include <wtf/CompletionHandler.h>
#include <wtf/Expected.h>

namespace JSC {
class JSGlobalObject;
}

namespace WebCore {

class DOMPromise;
class FetchRequest;
class FetchResponse;
class ResourceResponse;

class FetchEvent final : public ExtendableEvent {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(FetchEvent);
public:
    struct Init : ExtendableEventInit {
        RefPtr<FetchRequest> request;
        String clientId;
        String resultingClientId;
        RefPtr<DOMPromise> handled;
    };

    WEBCORE_EXPORT static Ref<FetchEvent> createForTesting(ScriptExecutionContext&);

    static Ref<FetchEvent> create(JSC::JSGlobalObject& globalObject, const AtomString& type, Init&& initializer, IsTrusted isTrusted = IsTrusted::No)
    {
        return adoptRef(*new FetchEvent(globalObject, type, WTFMove(initializer), isTrusted));
    }
    ~FetchEvent();

    ExceptionOr<void> respondWith(Ref<DOMPromise>&&);

    using ResponseCallback = CompletionHandler<void(Expected<Ref<FetchResponse>, std::optional<ResourceError>>&&)>;
    WEBCORE_EXPORT void onResponse(ResponseCallback&&);

    FetchRequest& request() { return m_request.get(); }
    const String& clientId() const { return m_clientId; }
    const String& resultingClientId() const { return m_resultingClientId; }
    DOMPromise& handled() const { return m_handled.get(); }

    bool respondWithEntered() const { return m_respondWithEntered; }

    static ResourceError createResponseError(const URL&, const String&, ResourceError::IsSanitized = ResourceError::IsSanitized::No);

    using PreloadResponsePromise = DOMPromiseProxy<IDLAny>;
    PreloadResponsePromise& preloadResponse(ScriptExecutionContext&);

    void setNavigationPreloadIdentifier(FetchIdentifier);
    WEBCORE_EXPORT void navigationPreloadIsReady(ResourceResponse&&);
    WEBCORE_EXPORT void navigationPreloadFailed(ResourceError&&);

private:
    WEBCORE_EXPORT FetchEvent(JSC::JSGlobalObject&, const AtomString&, Init&&, IsTrusted);

    void promiseIsSettled();
    void processResponse(Expected<Ref<FetchResponse>, std::optional<ResourceError>>&&);
    void respondWithError(ResourceError&&);

    Ref<FetchRequest> m_request;
    String m_clientId;
    String m_resultingClientId;

    bool m_respondWithEntered { false };
    bool m_waitToRespond { false };
    bool m_respondWithError { false };
    RefPtr<DOMPromise> m_respondPromise;
    Ref<DOMPromise> m_handled;

    ResponseCallback m_onResponse;

    Markable<FetchIdentifier> m_navigationPreloadIdentifier;
    std::unique_ptr<PreloadResponsePromise> m_preloadResponsePromise;
};

inline void FetchEvent::setNavigationPreloadIdentifier(FetchIdentifier identifier)
{
    ASSERT(!m_navigationPreloadIdentifier);
    m_navigationPreloadIdentifier = identifier;
}

} // namespace WebCore
