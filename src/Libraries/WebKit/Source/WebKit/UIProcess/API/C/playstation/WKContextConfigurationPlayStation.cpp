/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 26, 2023.
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
#include "WKContextConfigurationPlayStation.h"

#include "APIProcessPoolConfiguration.h"
#include "WKAPICast.h"

void WKContextConfigurationSetWebProcessPath(WKContextConfigurationRef configuration, WKStringRef webProcessPath)
{
    WebKit::toImpl(configuration)->setWebProcessPath(WebKit::toImpl(webProcessPath)->string());
}

WKStringRef WKContextConfigurationCopyWebProcessPath(WKContextConfigurationRef configuration)
{
    return WebKit::toCopiedAPI(WebKit::toImpl(configuration)->webProcessPath());
}

void WKContextConfigurationSetNetworkProcessPath(WKContextConfigurationRef configuration, WKStringRef networkProcessPath)
{
    WebKit::toImpl(configuration)->setNetworkProcessPath(WebKit::toImpl(networkProcessPath)->string());
}

WKStringRef WKContextConfigurationCopyNetworkProcessPath(WKContextConfigurationRef configuration)
{
    return WebKit::toCopiedAPI(WebKit::toImpl(configuration)->networkProcessPath());
}

void WKContextConfigurationSetUserId(WKContextConfigurationRef configuration, int32_t userId)
{
    WebKit::toImpl(configuration)->setUserId(userId);
}

int32_t WKContextConfigurationGetUserId(WKContextConfigurationRef configuration)
{
    return WebKit::toImpl(configuration)->userId();
}
