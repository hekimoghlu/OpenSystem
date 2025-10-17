/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 18, 2024.
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
#include "WebGeolocationProvider.h"

#include "WKAPICast.h"
#include "WebGeolocationManagerProxy.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(WebGeolocationProvider);

WebGeolocationProvider::WebGeolocationProvider(const WKGeolocationProviderBase* provider)
{
    initialize(provider);
}

void WebGeolocationProvider::startUpdating(WebGeolocationManagerProxy& geolocationManager)
{
    if (!m_client.startUpdating)
        return;

    m_client.startUpdating(toAPI(&geolocationManager), m_client.base.clientInfo);
}

void WebGeolocationProvider::stopUpdating(WebGeolocationManagerProxy& geolocationManager)
{
    if (!m_client.stopUpdating)
        return;

    m_client.stopUpdating(toAPI(&geolocationManager), m_client.base.clientInfo);
}

void WebGeolocationProvider::setEnableHighAccuracy(WebGeolocationManagerProxy& geolocationManager, bool enabled)
{
    if (!m_client.setEnableHighAccuracy)
        return;

    m_client.setEnableHighAccuracy(toAPI(&geolocationManager), enabled, m_client.base.clientInfo);
}

} // namespace WebKit
