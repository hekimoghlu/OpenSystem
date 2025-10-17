/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 2, 2023.
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

#if ENABLE(IMAGE_ANALYSIS) || HAVE(VISION)

#import <pal/spi/cocoa/VisionKitCoreSPI.h>
#import <wtf/CompletionHandler.h>
#import <wtf/RetainPtr.h>

OBJC_CLASS NSData;

#if ENABLE(IMAGE_ANALYSIS_ENHANCEMENTS)
using CocoaImageAnalysis = VKCImageAnalysis;
using CocoaImageAnalyzer = VKCImageAnalyzer;
using CocoaImageAnalyzerRequest = VKCImageAnalyzerRequest;
#elif ENABLE(IMAGE_ANALYSIS)
using CocoaImageAnalysis = VKImageAnalysis;
using CocoaImageAnalyzer = VKImageAnalyzer;
using CocoaImageAnalyzerRequest = VKImageAnalyzerRequest;
#endif

namespace WebCore {
struct TextRecognitionResult;
}

namespace WebKit {

#if ENABLE(IMAGE_ANALYSIS)

bool isLiveTextAvailableAndEnabled();
bool languageIdentifierSupportsLiveText(NSString *);

WebCore::TextRecognitionResult makeTextRecognitionResult(CocoaImageAnalysis *);

RetainPtr<CocoaImageAnalyzer> createImageAnalyzer();
RetainPtr<CocoaImageAnalyzerRequest> createImageAnalyzerRequest(CGImageRef, VKAnalysisTypes);

#if ENABLE(IMAGE_ANALYSIS_ENHANCEMENTS)
void requestVisualTranslation(CocoaImageAnalyzer *, NSURL *, const String& source, const String& target, CGImageRef, CompletionHandler<void(WebCore::TextRecognitionResult&&)>&&);
void requestBackgroundRemoval(CGImageRef, CompletionHandler<void(CGImageRef)>&&);

constexpr VKAnalysisTypes analysisTypesForElementFullscreenVideo()
{
    return VKAnalysisTypeText
#if ENABLE(IMAGE_ANALYSIS_FOR_MACHINE_READABLE_CODES)
        | VKAnalysisTypeMachineReadableCode
#endif
#if HAVE(VK_IMAGE_ANALYSIS_TYPE_IMAGE_SEGMENTATION)
        | VKAnalysisTypeImageSegmentation
#endif
        | VKAnalysisTypeAppClip;
}

constexpr VKAnalysisTypes analysisTypesForFullscreenVideo()
{
    return analysisTypesForElementFullscreenVideo() | VKAnalysisTypeVisualSearch;
}

std::pair<RetainPtr<NSData>, RetainPtr<CFStringRef>> imageDataForRemoveBackground(CGImageRef, const String& sourceMIMEType);

#if PLATFORM(IOS_FAMILY)
using PlatformImageAnalysisObject = VKCImageAnalysisInteraction;
#else
using PlatformImageAnalysisObject = VKCImageAnalysisOverlayView;
#endif
void prepareImageAnalysisForOverlayView(PlatformImageAnalysisObject *);
#endif // ENABLE(IMAGE_ANALYSIS_ENHANCEMENTS)

#endif // ENABLE(IMAGE_ANALYSIS)

#if HAVE(VISION)
void requestPayloadForQRCode(CGImageRef, CompletionHandler<void(NSString *)>&&);
#endif

}

#endif // ENABLE(IMAGE_ANALYSIS) || HAVE(VISION)
