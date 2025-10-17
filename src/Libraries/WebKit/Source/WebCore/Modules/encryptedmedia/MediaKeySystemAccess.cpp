/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 10, 2023.
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
#include "MediaKeySystemAccess.h"

#if ENABLE(ENCRYPTED_MEDIA)

#include "CDM.h"
#include "CDMInstance.h"
#include "Document.h"
#include "EventLoop.h"
#include "JSDOMPromiseDeferred.h"
#include "JSMediaKeys.h"
#include "MediaKeys.h"
#include "MediaKeysRequirement.h"

namespace WebCore {

Ref<MediaKeySystemAccess> MediaKeySystemAccess::create(Document& document, const String& keySystem, MediaKeySystemConfiguration&& configuration, Ref<CDM>&& implementation)
{
    auto access = adoptRef(*new MediaKeySystemAccess(document, keySystem, WTFMove(configuration), WTFMove(implementation)));
    access->suspendIfNeeded();
    return access;
}

MediaKeySystemAccess::MediaKeySystemAccess(Document& document, const String& keySystem, MediaKeySystemConfiguration&& configuration, Ref<CDM>&& implementation)
    : ActiveDOMObject(document)
    , m_keySystem(keySystem)
    , m_configuration(new MediaKeySystemConfiguration(WTFMove(configuration)))
    , m_implementation(WTFMove(implementation))
{
}

MediaKeySystemAccess::~MediaKeySystemAccess() = default;

void MediaKeySystemAccess::createMediaKeys(Document& document, Ref<DeferredPromise>&& promise)
{
    // https://w3c.github.io/encrypted-media/#dom-mediakeysystemaccess-createmediakeys
    // W3C Editor's Draft 09 November 2016

    // When this method is invoked, the user agent must run the following steps:
    // 1. Let promise be a new promise.
    // 2. Run the following steps in parallel:
    queueTaskKeepingObjectAlive(*this, TaskSource::MediaElement, [this, weakThis = WeakPtr { *this }, weakDocument = WeakPtr<Document, WeakPtrImplWithEventTargetData> { document }, promise = WTFMove(promise)]() mutable {
        RefPtr protectedThis = weakThis.get();
        if (!protectedThis)
            return;

        // 2.1. Let configuration be the value of this object's configuration value.
        // 2.2. Let use distinctive identifier be true if the value of configuration's distinctiveIdentifier member is "required" and false otherwise.
        bool useDistinctiveIdentifier = m_configuration->distinctiveIdentifier == MediaKeysRequirement::Required;

        // 2.3. Let persistent state allowed be true if the value of configuration's persistentState member is "required" and false otherwise.
        bool persistentStateAllowed = m_configuration->persistentState == MediaKeysRequirement::Required;

        // 2.4. Load and initialize the Key System implementation represented by this object's cdm implementation value if necessary.
        m_implementation->loadAndInitialize();

        // 2.5. Let instance be a new instance of the Key System implementation represented by this object's cdm implementation value.
        auto instance = m_implementation->createInstance();
        if (!instance) {
            promise->reject(ExceptionCode::InvalidStateError);
            return;
        }

        // 2.6. Initialize instance to enable, disable and/or select Key System features using configuration.
        // 2.7. If use distinctive identifier is false, prevent instance from using Distinctive Identifier(s) and Distinctive Permanent Identifier(s).
        // 2.8. If persistent state allowed is false, prevent instance from persisting any state related to the application or origin of this object's Document.
        auto allowDistinctiveIdentifiers = useDistinctiveIdentifier ? CDMInstance::AllowDistinctiveIdentifiers::Yes : CDMInstance::AllowDistinctiveIdentifiers::No;
        auto allowPersistentState = persistentStateAllowed ? CDMInstance::AllowPersistentState::Yes : CDMInstance::AllowPersistentState::No;

        instance->initializeWithConfiguration(*m_configuration, allowDistinctiveIdentifiers, allowPersistentState, [weakDocument = WTFMove(weakDocument), sessionTypes = m_configuration->sessionTypes, implementation = m_implementation.copyRef(), useDistinctiveIdentifier, persistentStateAllowed, instance = instance.releaseNonNull(), promise = WTFMove(promise)] (auto successValue) mutable {
            if (successValue == CDMInstance::Failed || !weakDocument) {
                promise->reject(ExceptionCode::NotAllowedError);
                return;
            }

            // 2.9. If any of the preceding steps failed, reject promise with a new DOMException whose name is the appropriate error name.
            // 2.10. Let media keys be a new MediaKeys object, and initialize it as follows:
            // 2.10.1. Let the use distinctive identifier value be use distinctive identifier.
            // 2.10.2. Let the persistent state allowed value be persistent state allowed.
            // 2.10.3. Let the supported session types value be be the value of configuration's sessionTypes member.
            // 2.10.4. Let the cdm implementation value be this object's cdm implementation value.
            // 2.10.5. Let the cdm instance value be instance.
            auto mediaKeys = MediaKeys::create(*weakDocument, useDistinctiveIdentifier, persistentStateAllowed, sessionTypes, WTFMove(implementation), WTFMove(instance));

            // 2.11. Resolve promise with media keys.
            promise->resolveWithNewlyCreated<IDLInterface<MediaKeys>>(WTFMove(mediaKeys));
        });
    });

    // 3. Return promise.
}

} // namespace WebCore

#endif // ENABLE(ENCRYPTED_MEDIA)
