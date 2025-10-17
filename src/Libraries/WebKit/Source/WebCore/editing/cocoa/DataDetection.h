/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 13, 2024.
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

#if ENABLE(DATA_DETECTION)

#import "DataDetectorType.h"
#import "FloatRect.h"
#import "SimpleRange.h"
#import <wtf/OptionSet.h>

#import <wtf/RetainPtr.h>

#if HAVE(SECURE_ACTION_CONTEXT)
OBJC_CLASS DDSecureActionContext;
using WKDDActionContext = DDSecureActionContext;
#else
OBJC_CLASS DDActionContext;
using WKDDActionContext = DDActionContext;
#endif
OBJC_CLASS NSArray;
OBJC_CLASS NSDictionary;

typedef struct __DDResult *DDResultRef;
typedef struct __DDScanQuery *DDScanQueryRef;
typedef struct __DDScanner *DDScannerRef;

namespace WebCore {

class Document;
class HTMLDivElement;
class HTMLElement;
class HitTestResult;
class QualifiedName;
class LocalFrame;
struct TextRecognitionDataDetector;

struct DetectedItem {
    RetainPtr<WKDDActionContext> actionContext;
    FloatRect boundingBox;
    SimpleRange range;
};

class DataDetection {
public:
#if PLATFORM(MAC)
    WEBCORE_EXPORT static std::optional<DetectedItem> detectItemAroundHitTestResult(const HitTestResult&);
#endif
    WEBCORE_EXPORT static void detectContentInFrame(LocalFrame*, OptionSet<DataDetectorType>, std::optional<double>, CompletionHandler<void(NSArray *)>&&);
    WEBCORE_EXPORT static NSArray * detectContentInRange(const SimpleRange&, OptionSet<DataDetectorType>, std::optional<double> referenceDate);
    WEBCORE_EXPORT static std::optional<double> extractReferenceDate(NSDictionary *);
    WEBCORE_EXPORT static void removeDataDetectedLinksInDocument(Document&);
#if PLATFORM(IOS_FAMILY)
    WEBCORE_EXPORT static bool canBePresentedByDataDetectors(const URL&);
    WEBCORE_EXPORT static bool isDataDetectorLink(Element&);
    WEBCORE_EXPORT static String dataDetectorIdentifier(Element&);
    WEBCORE_EXPORT static bool canPresentDataDetectorsUIForElement(Element&);
    WEBCORE_EXPORT static bool requiresExtendedContext(Element&);
#endif
    WEBCORE_EXPORT static std::optional<std::pair<Ref<HTMLElement>, IntRect>> findDataDetectionResultElementInImageOverlay(const FloatPoint& location, const HTMLElement& imageOverlayHost);

#if ENABLE(IMAGE_ANALYSIS)
    static Ref<HTMLDivElement> createElementForImageOverlay(Document&, const TextRecognitionDataDetector&);
#endif

    static const String& dataDetectorURLProtocol();
    static bool isDataDetectorURL(const URL&);
    static bool isDataDetectorAttribute(const QualifiedName&);
    static bool isDataDetectorElement(const Element&);
};

} // namespace WebCore

#endif
