/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 8, 2023.
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
#include "BackgroundFetchFailureReason.h"
#include "BackgroundFetchInformation.h"
#include "BackgroundFetchResult.h"
#include "EventTarget.h"
#include "JSDOMPromiseDeferred.h"
#include <wtf/RefCounted.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class BackgroundFetchRecord;
class FetchRequest;
struct BackgroundFetchInformation;
struct BackgroundFetchRecordInformation;
struct CacheQueryOptions;

class BackgroundFetchRegistration final : public RefCounted<BackgroundFetchRegistration>, public EventTarget, public ActiveDOMObject {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(BackgroundFetchRegistration);
public:
    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    static Ref<BackgroundFetchRegistration> create(ScriptExecutionContext&, BackgroundFetchInformation&&);
    ~BackgroundFetchRegistration();

    static void updateIfExisting(ScriptExecutionContext&, const BackgroundFetchInformation&);

    const String& id() const { return m_information.identifier; }
    uint64_t uploadTotal() const { return m_information.uploadTotal; }
    uint64_t uploaded() const { return m_information.uploaded; }
    uint64_t downloadTotal() const { return m_information.downloadTotal; }
    uint64_t downloaded() const { return m_information.downloaded; }
    BackgroundFetchResult result() const { return m_information.result; }
    BackgroundFetchFailureReason failureReason() const { return m_information.failureReason; }
    bool recordsAvailable() const { return m_information.recordsAvailable; }

    using RequestInfo = std::variant<RefPtr<FetchRequest>, String>;

    void abort(ScriptExecutionContext&, DOMPromiseDeferred<IDLBoolean>&&);
    void match(ScriptExecutionContext&, RequestInfo&&, const CacheQueryOptions&, DOMPromiseDeferred<IDLInterface<BackgroundFetchRecord>>&&);
    void matchAll(ScriptExecutionContext&, std::optional<RequestInfo>&&, const CacheQueryOptions&, DOMPromiseDeferred<IDLSequence<IDLInterface<BackgroundFetchRecord>>>&&);

    void updateInformation(const BackgroundFetchInformation&);

private:
    BackgroundFetchRegistration(ScriptExecutionContext&, BackgroundFetchInformation&&);

    ServiceWorkerRegistrationIdentifier registrationIdentifier() const { return m_information.registrationIdentifier; }

    // EventTarget
    enum EventTargetInterfaceType eventTargetInterface() const final { return EventTargetInterfaceType::BackgroundFetchRegistration; }
    ScriptExecutionContext* scriptExecutionContext() const final { return ActiveDOMObject::scriptExecutionContext(); }
    void refEventTarget() final { ref(); }
    void derefEventTarget() final { deref(); }

    // ActiveDOMObject
    void stop() final;
    bool virtualHasPendingActivity() const final;

    BackgroundFetchInformation m_information;
};

} // namespace WebCore
