/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 2, 2022.
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
#include "WKGeolocationManager.h"

#include "WKAPICast.h"
#include "WebGeolocationManagerProxy.h"
#include "WebGeolocationPosition.h"
#include "WebGeolocationProvider.h"

using namespace WebKit;

WKTypeID WKGeolocationManagerGetTypeID()
{
    return toAPI(WebGeolocationManagerProxy::APIType);
}

void WKGeolocationManagerSetProvider(WKGeolocationManagerRef geolocationManagerRef, const WKGeolocationProviderBase* wkProvider)
{
    toImpl(geolocationManagerRef)->setProvider(makeUnique<WebGeolocationProvider>(wkProvider));
}

void WKGeolocationManagerProviderDidChangePosition(WKGeolocationManagerRef geolocationManagerRef, WKGeolocationPositionRef positionRef)
{
    toImpl(geolocationManagerRef)->providerDidChangePosition(toImpl(positionRef));
}

void WKGeolocationManagerProviderDidFailToDeterminePosition(WKGeolocationManagerRef geolocationManagerRef)
{
    toImpl(geolocationManagerRef)->providerDidFailToDeterminePosition();
}

void WKGeolocationManagerProviderDidFailToDeterminePositionWithErrorMessage(WKGeolocationManagerRef geolocationManagerRef, WKStringRef errorMessage)
{
    toImpl(geolocationManagerRef)->providerDidFailToDeterminePosition(toWTFString(errorMessage));
}
