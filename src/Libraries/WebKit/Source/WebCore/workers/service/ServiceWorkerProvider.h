/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 13, 2023.
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

#include "ServiceWorkerIdentifier.h"

namespace WebCore {

class SWClientConnection;
class SecurityOriginData;
class ServiceWorkerJob;

class WEBCORE_EXPORT ServiceWorkerProvider {
public:
    virtual ~ServiceWorkerProvider();

    static ServiceWorkerProvider& singleton();
    static void setSharedProvider(ServiceWorkerProvider&);

    virtual SWClientConnection& serviceWorkerConnection() = 0;
    virtual SWClientConnection* existingServiceWorkerConnection() = 0;
    virtual void terminateWorkerForTesting(ServiceWorkerIdentifier, CompletionHandler<void()>&&) = 0;

    void setMayHaveRegisteredServiceWorkers() { m_mayHaveRegisteredServiceWorkers = true; }

private:
    bool m_mayHaveRegisteredServiceWorkers { false };
};

} // namespace WebCore
