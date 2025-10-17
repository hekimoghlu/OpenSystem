/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 25, 2022.
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

#if ENABLE(ENCRYPTED_MEDIA)

#include "ActiveDOMObject.h"
#include "IDLTypes.h"
#include "JSDOMPromiseDeferredForward.h"
#include "MediaKeySystemRequestIdentifier.h"
#include <wtf/CompletionHandler.h>
#include <wtf/Forward.h>
#include <wtf/Identified.h>
#include <wtf/ObjectIdentifier.h>
#include <wtf/UniqueRef.h>

namespace WebCore {

class SecurityOrigin;
class MediaKeySystem;

template <typename IDLType> class DOMPromiseDeferred;

class MediaKeySystemRequest : public RefCounted<MediaKeySystemRequest>, public ActiveDOMObject, public Identified<MediaKeySystemRequestIdentifier> {
public:
    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    WEBCORE_EXPORT static Ref<MediaKeySystemRequest> create(Document&, const String& keySystem, Ref<DeferredPromise>&&);
    virtual ~MediaKeySystemRequest();

    void setAllowCallback(CompletionHandler<void(Ref<DeferredPromise>&&)>&& callback) { m_allowCompletionHandler = WTFMove(callback); }
    WEBCORE_EXPORT void start();

    WEBCORE_EXPORT void allow();
    WEBCORE_EXPORT void deny(const String& errorMessage = emptyString());

    WEBCORE_EXPORT SecurityOrigin* topLevelDocumentOrigin() const;
    WEBCORE_EXPORT Document* document() const;

    const String keySystem() const { return m_keySystem; }

private:
    MediaKeySystemRequest(Document&, const String& keySystem, Ref<DeferredPromise>&&);

    // ActiveDOMObject.
    void stop() final;

    String m_keySystem;
    Ref<DeferredPromise> m_promise;

    CompletionHandler<void(Ref<DeferredPromise>&&)> m_allowCompletionHandler;
};

} // namespace WebCore

#endif // ENABLE(ENCRYPTED_MEDIA)
