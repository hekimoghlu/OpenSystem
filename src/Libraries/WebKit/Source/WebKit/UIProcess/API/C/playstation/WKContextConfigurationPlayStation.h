/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 30, 2022.
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

#include <WebKit/WKBase.h>
#include <WebKit/WKContextConfigurationRef.h>

#ifdef __cplusplus
extern "C" {
#endif

WK_EXPORT void WKContextConfigurationSetWebProcessPath(WKContextConfigurationRef configuration, WKStringRef webProcessPath);
WK_EXPORT WKStringRef WKContextConfigurationCopyWebProcessPath(WKContextConfigurationRef configuration);

WK_EXPORT void WKContextConfigurationSetNetworkProcessPath(WKContextConfigurationRef configuration, WKStringRef networkProcessPath);
WK_EXPORT WKStringRef WKContextConfigurationCopyNetworkProcessPath(WKContextConfigurationRef configuration);

WK_EXPORT void WKContextConfigurationSetUserId(WKContextConfigurationRef configuration, int32_t userId);
WK_EXPORT int32_t WKContextConfigurationGetUserId(WKContextConfigurationRef configuration);

#ifdef __cplusplus
}
#endif
