/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 28, 2022.
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

#include <WebCore/SecurityOriginData.h>
#include <wtf/CompletionHandler.h>
#include <wtf/Vector.h>

#if PLATFORM(COCOA)
OBJC_CLASS NSString;
#endif

namespace WebCore {
class SecurityOrigin;
}

namespace WebKit {

class WebPageProxy;

enum class MediaPermissionType : uint8_t {
    Audio = 1 << 0,
    Video = 1 << 1
};

enum class MediaPermissionResult {
    Denied,
    Granted,
    Unknown
};

enum class MediaPermissionReason {
    Camera,
    CameraAndMicrophone,
    Microphone,
    DeviceOrientation,
    Geolocation,
    SpeechRecognition,
    ScreenCapture
};

#if PLATFORM(COCOA)
bool checkSandboxRequirementForType(MediaPermissionType);
bool checkUsageDescriptionStringForType(MediaPermissionType);
bool checkUsageDescriptionStringForSpeechRecognition();

NSString *applicationVisibleNameFromOrigin(const WebCore::SecurityOriginData&);
NSString *applicationVisibleName();
void alertForPermission(WebPageProxy&, MediaPermissionReason, const WebCore::SecurityOriginData&, CompletionHandler<void(bool)>&&);

void requestAVCaptureAccessForType(MediaPermissionType, CompletionHandler<void(bool authorized)>&&);
MediaPermissionResult checkAVCaptureAccessForType(MediaPermissionType);
#endif

#if HAVE(SPEECHRECOGNIZER)
void requestSpeechRecognitionAccess(CompletionHandler<void(bool authorized)>&&);
MediaPermissionResult checkSpeechRecognitionServiceAccess();
bool checkSpeechRecognitionServiceAvailability(const String& localeIdentifier);
#endif

} // namespace WebKit
