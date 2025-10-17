/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 23, 2023.
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

#if HAVE(MEDIA_ACCESSIBILITY_FRAMEWORK)

#include <CoreText/CoreText.h>
#include <ImageIO/CGImageSource.h>
#include <MediaAccessibility/MediaAccessibility.h>
#include <wtf/SoftLinking.h>

#if COMPILER(MSVC)
// See https://msdn.microsoft.com/en-us/library/35bhkfb6.aspx
#pragma warning(disable: 4273)
#endif

SOFT_LINK_FRAMEWORK_FOR_SOURCE(WebCore, MediaAccessibility)

SOFT_LINK_FUNCTION_FOR_SOURCE(WebCore, MediaAccessibility, MACaptionAppearanceGetDisplayType, MACaptionAppearanceDisplayType, (MACaptionAppearanceDomain domain), (domain))
SOFT_LINK_FUNCTION_FOR_SOURCE(WebCore, MediaAccessibility, MACaptionAppearanceSetDisplayType, void, (MACaptionAppearanceDomain domain, MACaptionAppearanceDisplayType displayType), (domain, displayType))
SOFT_LINK_FUNCTION_FOR_SOURCE(WebCore, MediaAccessibility, MACaptionAppearanceCopyForegroundColor, CGColorRef, (MACaptionAppearanceDomain domain, MACaptionAppearanceBehavior *behavior), (domain, behavior))
SOFT_LINK_FUNCTION_FOR_SOURCE(WebCore, MediaAccessibility, MACaptionAppearanceCopyBackgroundColor, CGColorRef, (MACaptionAppearanceDomain domain, MACaptionAppearanceBehavior *behavior), (domain, behavior))
SOFT_LINK_FUNCTION_FOR_SOURCE(WebCore, MediaAccessibility, MACaptionAppearanceCopyWindowColor, CGColorRef, (MACaptionAppearanceDomain domain, MACaptionAppearanceBehavior *behavior), (domain, behavior))
SOFT_LINK_FUNCTION_FOR_SOURCE(WebCore, MediaAccessibility, MACaptionAppearanceGetForegroundOpacity, CGFloat, (MACaptionAppearanceDomain domain, MACaptionAppearanceBehavior *behavior), (domain, behavior))
SOFT_LINK_FUNCTION_FOR_SOURCE(WebCore, MediaAccessibility, MACaptionAppearanceGetBackgroundOpacity, CGFloat, (MACaptionAppearanceDomain domain, MACaptionAppearanceBehavior *behavior), (domain, behavior))
SOFT_LINK_FUNCTION_FOR_SOURCE(WebCore, MediaAccessibility, MACaptionAppearanceGetWindowOpacity, CGFloat, (MACaptionAppearanceDomain domain, MACaptionAppearanceBehavior *behavior), (domain, behavior))
SOFT_LINK_FUNCTION_FOR_SOURCE(WebCore, MediaAccessibility, MACaptionAppearanceGetWindowRoundedCornerRadius, CGFloat, (MACaptionAppearanceDomain domain, MACaptionAppearanceBehavior *behavior), (domain, behavior))
SOFT_LINK_FUNCTION_FOR_SOURCE(WebCore, MediaAccessibility, MACaptionAppearanceCopyFontDescriptorForStyle, CTFontDescriptorRef, (MACaptionAppearanceDomain domain,  MACaptionAppearanceBehavior *behavior, MACaptionAppearanceFontStyle fontStyle), (domain, behavior, fontStyle))
SOFT_LINK_FUNCTION_FOR_SOURCE(WebCore, MediaAccessibility, MACaptionAppearanceGetRelativeCharacterSize, CGFloat, (MACaptionAppearanceDomain domain, MACaptionAppearanceBehavior *behavior), (domain, behavior))
SOFT_LINK_FUNCTION_FOR_SOURCE(WebCore, MediaAccessibility, MACaptionAppearanceGetTextEdgeStyle, MACaptionAppearanceTextEdgeStyle, (MACaptionAppearanceDomain domain, MACaptionAppearanceBehavior *behavior), (domain, behavior))
SOFT_LINK_FUNCTION_FOR_SOURCE(WebCore, MediaAccessibility, MACaptionAppearanceAddSelectedLanguage, bool, (MACaptionAppearanceDomain domain, CFStringRef language), (domain, language));
SOFT_LINK_FUNCTION_FOR_SOURCE(WebCore, MediaAccessibility, MACaptionAppearanceCopySelectedLanguages, CFArrayRef, (MACaptionAppearanceDomain domain), (domain));
SOFT_LINK_FUNCTION_FOR_SOURCE(WebCore, MediaAccessibility, MACaptionAppearanceCopyPreferredCaptioningMediaCharacteristics,  CFArrayRef, (MACaptionAppearanceDomain domain), (domain));
SOFT_LINK_FUNCTION_MAY_FAIL_FOR_SOURCE(WebCore, MediaAccessibility, MAAudibleMediaCopyPreferredCharacteristics, CFArrayRef, (), ());
SOFT_LINK_FUNCTION_MAY_FAIL_FOR_SOURCE(WebCore, MediaAccessibility, MACaptionAppearanceCopyFontDescriptorWithStrokeForStyle, CTFontDescriptorRef, (MACaptionAppearanceDomain domain, MACaptionAppearanceBehavior *behavior, MACaptionAppearanceFontStyle fontStyle, CFStringRef trackLanguage, CGFloat darwingPointSize, CGFloat *strokeWidthPt), (domain, behavior, fontStyle, trackLanguage, darwingPointSize, strokeWidthPt));
SOFT_LINK_CONSTANT_FOR_SOURCE_WITH_EXPORT(WebCore, MediaAccessibility, kMAXCaptionAppearanceSettingsChangedNotification, CFStringRef, WEBCORE_EXPORT)
SOFT_LINK_CONSTANT_MAY_FAIL_FOR_SOURCE(WebCore, MediaAccessibility, kMAAudibleMediaSettingsChangedNotification, CFStringRef)
SOFT_LINK_FUNCTION_MAY_FAIL_FOR_SOURCE(WebCore, MediaAccessibility, MAImageCaptioningCopyCaptionWithSource, CFStringRef, (CGImageSourceRef imageSource, CFErrorRef *error), (imageSource, error))
SOFT_LINK_FUNCTION_FOR_SOURCE(WebCore, MediaAccessibility, MAAudibleMediaPrefCopyPreferDescriptiveVideo, CFBooleanRef, (), ())
SOFT_LINK_FUNCTION_MAY_FAIL_FOR_SOURCE(WebCore, MediaAccessibility, MACaptionAppearanceIsCustomized, bool, (MACaptionAppearanceDomain domain), (domain));
#endif // HAVE(MEDIA_ACCESSIBILITY_FRAMEWORK)
