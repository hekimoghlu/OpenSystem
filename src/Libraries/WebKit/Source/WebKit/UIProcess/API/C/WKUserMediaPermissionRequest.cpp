/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 12, 2025.
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
#include "WKUserMediaPermissionRequest.h"

#include "UserMediaPermissionRequestProxy.h"
#include "WKAPICast.h"
#include "WKArray.h"
#include "WKMutableArray.h"
#include "WKString.h"

using namespace WebKit;

WKTypeID WKUserMediaPermissionRequestGetTypeID()
{
    return toAPI(UserMediaPermissionRequestProxy::APIType);
}


void WKUserMediaPermissionRequestAllow(WKUserMediaPermissionRequestRef userMediaPermissionRequestRef, WKStringRef audioDeviceUID, WKStringRef videoDeviceUID)
{
    toImpl(userMediaPermissionRequestRef)->allow(toWTFString(audioDeviceUID), toWTFString(videoDeviceUID));
}

static UserMediaPermissionRequestProxy::UserMediaAccessDenialReason toWK(UserMediaPermissionRequestDenialReason reason)
{
    switch (reason) {
    case kWKNoConstraints:
        return UserMediaPermissionRequestProxy::UserMediaAccessDenialReason::NoConstraints;
        break;
    case kWKUserMediaDisabled:
        return UserMediaPermissionRequestProxy::UserMediaAccessDenialReason::UserMediaDisabled;
        break;
    case kWKNoCaptureDevices:
        return UserMediaPermissionRequestProxy::UserMediaAccessDenialReason::NoCaptureDevices;
        break;
    case kWKInvalidConstraint:
        return UserMediaPermissionRequestProxy::UserMediaAccessDenialReason::InvalidConstraint;
        break;
    case kWKHardwareError:
        return UserMediaPermissionRequestProxy::UserMediaAccessDenialReason::HardwareError;
        break;
    case kWKPermissionDenied:
        return UserMediaPermissionRequestProxy::UserMediaAccessDenialReason::PermissionDenied;
        break;
    case kWKOtherFailure:
        return UserMediaPermissionRequestProxy::UserMediaAccessDenialReason::OtherFailure;
        break;
    }

    ASSERT_NOT_REACHED();
    return UserMediaPermissionRequestProxy::UserMediaAccessDenialReason::OtherFailure;
    
}

void WKUserMediaPermissionRequestDeny(WKUserMediaPermissionRequestRef userMediaPermissionRequestRef, UserMediaPermissionRequestDenialReason reason)
{
    toImpl(userMediaPermissionRequestRef)->deny(toWK(reason));
}

WKArrayRef WKUserMediaPermissionRequestVideoDeviceUIDs(WKUserMediaPermissionRequestRef userMediaPermissionRef)
{
    WKMutableArrayRef array = WKMutableArrayCreate();
#if ENABLE(MEDIA_STREAM)
    for (auto& deviceUID : toImpl(userMediaPermissionRef)->videoDeviceUIDs())
        WKArrayAppendItem(array, toAPI(API::String::create(deviceUID).ptr()));
#endif
    return array;
}

WKArrayRef WKUserMediaPermissionRequestAudioDeviceUIDs(WKUserMediaPermissionRequestRef userMediaPermissionRef)
{
    WKMutableArrayRef array = WKMutableArrayCreate();
#if ENABLE(MEDIA_STREAM)
    for (auto& deviceUID : toImpl(userMediaPermissionRef)->audioDeviceUIDs())
        WKArrayAppendItem(array, toAPI(API::String::create(deviceUID).ptr()));
#endif
    return array;
}

bool WKUserMediaPermissionRequestRequiresCameraCapture(WKUserMediaPermissionRequestRef userMediaPermissionRequestRef)
{
    return toImpl(userMediaPermissionRequestRef)->requiresVideoCapture();
}

bool WKUserMediaPermissionRequestRequiresDisplayCapture(WKUserMediaPermissionRequestRef userMediaPermissionRequestRef)
{
    return toImpl(userMediaPermissionRequestRef)->requiresDisplayCapture();
}

bool WKUserMediaPermissionRequestRequiresMicrophoneCapture(WKUserMediaPermissionRequestRef userMediaPermissionRequestRef)
{
    return toImpl(userMediaPermissionRequestRef)->requiresAudioCapture();
}
