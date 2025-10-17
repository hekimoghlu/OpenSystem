/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 9, 2024.
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
#import "config.h"

#import "ArgumentCodersCocoa.h"
#import <CoreText/CoreText.h>
#import <WebCore/AttributedString.h>
#import <WebCore/DataDetectorElementInfo.h>
#import <WebCore/DictionaryPopupInfo.h>
#import <WebCore/Font.h>
#import <WebCore/FontAttributes.h>
#import <WebCore/FontCustomPlatformData.h>
#import <WebCore/ResourceRequest.h>
#import <WebCore/SharedMemory.h>
#import <WebCore/TextRecognitionResult.h>
#import <pal/spi/cf/CoreTextSPI.h>

#if PLATFORM(IOS_FAMILY)
#import <UIKit/UIFont.h>
#endif

#if ENABLE(WIRELESS_PLAYBACK_TARGET)
#import <WebCore/MediaPlaybackTargetContext.h>
#import <objc/runtime.h>
#endif

#if USE(APPLE_INTERNAL_SDK)
#include <WebKitAdditions/WebCoreArgumentCodersCocoaAdditions.mm>
#endif

#if USE(AVFOUNDATION)
#import <wtf/MachSendRight.h>
#endif

#if ENABLE(APPLE_PAY)
#import <pal/cocoa/PassKitSoftLink.h>
#endif

#if ENABLE(WIRELESS_PLAYBACK_TARGET)
#import <pal/cocoa/AVFoundationSoftLink.h>
#endif

#if ENABLE(DATA_DETECTION)
#import <pal/cocoa/DataDetectorsCoreSoftLink.h>
#endif

#if USE(AVFOUNDATION)
#import <WebCore/CoreVideoSoftLink.h>
#endif

#import <pal/cocoa/VisionKitCoreSoftLink.h>

namespace IPC {

} // namespace IPC
