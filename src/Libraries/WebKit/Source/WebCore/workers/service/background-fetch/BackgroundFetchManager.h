/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 23, 2021.
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

#include "BackgroundFetchRegistration.h"
#include "JSDOMPromiseDeferred.h"
#include "ServiceWorkerTypes.h"
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

struct BackgroundFetchInformation;
class BackgroundFetchRegistration;
struct BackgroundFetchRegistrationData;
struct BackgroundFetchOptions;
class FetchRequest;
class ServiceWorkerRegistration;

class BackgroundFetchManager : public RefCountedAndCanMakeWeakPtr<BackgroundFetchManager> {
public:
    static Ref<BackgroundFetchManager> create(ServiceWorkerRegistration& registration) { return adoptRef(*new BackgroundFetchManager(registration)); }
    ~BackgroundFetchManager();

    using RequestInfo = std::variant<RefPtr<FetchRequest>, String>;
    using Requests = std::variant<RefPtr<FetchRequest>, String, Vector<RequestInfo>>;
    void fetch(ScriptExecutionContext&, const String&, Requests&&, BackgroundFetchOptions&&, DOMPromiseDeferred<IDLInterface<BackgroundFetchRegistration>>&&);
    void get(ScriptExecutionContext&, const String&, DOMPromiseDeferred<IDLNullable<IDLInterface<BackgroundFetchRegistration>>>&&);
    void getIds(ScriptExecutionContext&, DOMPromiseDeferred<IDLSequence<IDLDOMString>>&&);

    RefPtr<BackgroundFetchRegistration> existingBackgroundFetchRegistration(const String& identifier) { return m_backgroundFetchRegistrations.get(identifier); }
    Ref<BackgroundFetchRegistration> backgroundFetchRegistrationInstance(ScriptExecutionContext&, BackgroundFetchInformation&&);

private:
    explicit BackgroundFetchManager(ServiceWorkerRegistration&);

    ServiceWorkerRegistrationIdentifier m_identifier;
    HashMap<String, Ref<BackgroundFetchRegistration>> m_backgroundFetchRegistrations;
};

} // namespace WebCore
