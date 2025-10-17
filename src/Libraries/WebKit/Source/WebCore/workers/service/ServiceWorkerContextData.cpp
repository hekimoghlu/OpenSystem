/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 9, 2021.
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
#include "ServiceWorkerContextData.h"
#include <wtf/CrossThreadCopier.h>

namespace WebCore {

ServiceWorkerContextData ServiceWorkerContextData::isolatedCopy() const &
{
    return {
        jobDataIdentifier,
        registration.isolatedCopy(),
        serviceWorkerIdentifier,
        script.isolatedCopy(),
        certificateInfo.isolatedCopy(),
        contentSecurityPolicy.isolatedCopy(),
        crossOriginEmbedderPolicy.isolatedCopy(),
        referrerPolicy.isolatedCopy(),
        scriptURL.isolatedCopy(),
        workerType,
        loadedFromDisk,
        lastNavigationWasAppInitiated,
        crossThreadCopy(scriptResourceMap),
        serviceWorkerPageIdentifier,
        crossThreadCopy(navigationPreloadState),
    };
}

ServiceWorkerContextData ServiceWorkerContextData::isolatedCopy() &&
{
    return {
        jobDataIdentifier,
        WTFMove(registration).isolatedCopy(),
        serviceWorkerIdentifier,
        WTFMove(script).isolatedCopy(),
        WTFMove(certificateInfo).isolatedCopy(),
        WTFMove(contentSecurityPolicy).isolatedCopy(),
        WTFMove(crossOriginEmbedderPolicy).isolatedCopy(),
        WTFMove(referrerPolicy).isolatedCopy(),
        WTFMove(scriptURL).isolatedCopy(),
        workerType,
        loadedFromDisk,
        lastNavigationWasAppInitiated,
        crossThreadCopy(WTFMove(scriptResourceMap)),
        serviceWorkerPageIdentifier,
        crossThreadCopy(WTFMove(navigationPreloadState))
    };
}

} // namespace WebCore
