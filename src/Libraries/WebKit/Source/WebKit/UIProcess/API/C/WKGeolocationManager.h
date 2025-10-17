/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 26, 2025.
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
#ifndef WKGeolocationManager_h
#define WKGeolocationManager_h

#include <WebKit/WKBase.h>

#ifdef __cplusplus
extern "C" {
#endif

// Provider.
typedef void (*WKGeolocationProviderStartUpdatingCallback)(WKGeolocationManagerRef geolocationManager, const void* clientInfo);
typedef void (*WKGeolocationProviderStopUpdatingCallback)(WKGeolocationManagerRef geolocationManager, const void* clientInfo);
typedef void (*WKGeolocationProviderSetEnableHighAccuracyCallback)(WKGeolocationManagerRef geolocationManager, bool enabled, const void* clientInfo);

typedef struct WKGeolocationProviderBase {
    int                                                                 version;
    const void *                                                        clientInfo;
} WKGeolocationProviderBase;

typedef struct WKGeolocationProviderV0 {
    WKGeolocationProviderBase                                           base;

    // Version 0.
    WKGeolocationProviderStartUpdatingCallback                          startUpdating;
    WKGeolocationProviderStopUpdatingCallback                           stopUpdating;
} WKGeolocationProviderV0;

typedef struct WKGeolocationProviderV1 {
    WKGeolocationProviderBase                                           base;

    // Version 0.
    WKGeolocationProviderStartUpdatingCallback                          startUpdating;
    WKGeolocationProviderStopUpdatingCallback                           stopUpdating;

    // Version 1.
    WKGeolocationProviderSetEnableHighAccuracyCallback                  setEnableHighAccuracy;
} WKGeolocationProviderV1;


WK_EXPORT WKTypeID WKGeolocationManagerGetTypeID(void);

WK_EXPORT void WKGeolocationManagerSetProvider(WKGeolocationManagerRef geolocationManager, const WKGeolocationProviderBase* provider);

WK_EXPORT void WKGeolocationManagerProviderDidChangePosition(WKGeolocationManagerRef geolocationManager, WKGeolocationPositionRef position);
WK_EXPORT void WKGeolocationManagerProviderDidFailToDeterminePosition(WKGeolocationManagerRef geolocationManager);

WK_EXPORT void WKGeolocationManagerProviderDidFailToDeterminePositionWithErrorMessage(WKGeolocationManagerRef geolocationManager, WKStringRef errorMessage);

#ifdef __cplusplus
}
#endif

#endif /* WKGeolocationManager_h */
