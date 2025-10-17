/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 23, 2025.
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

#include <pal/spi/cocoa/DataDetectorsCoreSPI.h>
#include <wtf/SoftLinking.h>

SOFT_LINK_FRAMEWORK_FOR_HEADER(PAL, DataDetectorsCore);

SOFT_LINK_CLASS_FOR_HEADER(PAL, DDScannerResult)

#if PLATFORM(MAC)
SOFT_LINK_CONSTANT_FOR_HEADER(PAL, DataDetectorsCore, DDBinderPhoneNumberKey, CFStringRef)
#elif PLATFORM(IOS_FAMILY)
SOFT_LINK_FUNCTION_FOR_HEADER(PAL, DataDetectorsCore, DDResultGetRange, CFRange, (DDResultRef result), (result))
SOFT_LINK_FUNCTION_FOR_HEADER(PAL, DataDetectorsCore, DDResultGetType, CFStringRef, (DDResultRef result), (result))
SOFT_LINK_FUNCTION_FOR_HEADER(PAL, DataDetectorsCore, DDResultGetCategory, DDResultCategory, (DDResultRef result), (result))
SOFT_LINK_FUNCTION_FOR_HEADER(PAL, DataDetectorsCore, DDScanQueryAddTextFragment, void, (DDScanQueryRef query, CFStringRef fragment, CFRange range, void *identifier, DDTextFragmentMode mode, DDTextCoalescingType type), (query, fragment, range, identifier, mode, type))
SOFT_LINK_FUNCTION_FOR_HEADER(PAL, DataDetectorsCore, DDScanQueryAddSeparator, void, (DDScanQueryRef query, DDTextCoalescingType type), (query, type))
SOFT_LINK_FUNCTION_FOR_HEADER(PAL, DataDetectorsCore, DDScanQueryAddLineBreak, void, (DDScanQueryRef query), (query))
SOFT_LINK_FUNCTION_FOR_HEADER(PAL, DataDetectorsCore, DDScanQueryGetFragmentMetaData, void *, (DDScanQueryRef query, CFIndex queryIndex), (query, queryIndex))
SOFT_LINK_FUNCTION_FOR_HEADER(PAL, DataDetectorsCore, DDResultHasProperties, bool, (DDResultRef result, CFIndex propertySet), (result, propertySet))
SOFT_LINK_FUNCTION_FOR_HEADER(PAL, DataDetectorsCore, DDResultGetSubResults, CFArrayRef, (DDResultRef result), (result))
SOFT_LINK_FUNCTION_FOR_HEADER(PAL, DataDetectorsCore, DDResultGetQueryRangeForURLification, DDQueryRange, (DDResultRef result), (result))
SOFT_LINK_FUNCTION_FOR_HEADER(PAL, DataDetectorsCore, DDURLStringForResult, NSString *, (DDResultRef currentResult, NSString * resultIdentifier, DDURLifierPhoneNumberDetectionTypes includingTelGroups, NSDate * referenceDate, NSTimeZone * referenceTimeZone), (currentResult, resultIdentifier, includingTelGroups, referenceDate, referenceTimeZone))
SOFT_LINK_FUNCTION_FOR_HEADER(PAL, DataDetectorsCore, DDURLTapAndHoldSchemes, NSArray *, (), ())
SOFT_LINK_FUNCTION_FOR_HEADER(PAL, DataDetectorsCore, DDShouldImmediatelyShowActionSheetForURL, BOOL, (NSURL *url), (url))
SOFT_LINK_FUNCTION_FOR_HEADER(PAL, DataDetectorsCore, DDShouldImmediatelyShowActionSheetForResult, BOOL, (DDResultRef result), (result))
SOFT_LINK_FUNCTION_FOR_HEADER(PAL, DataDetectorsCore, DDShouldUseLightLinksForResult, BOOL, (DDResultRef result, BOOL extractedFromSignature), (result, extractedFromSignature))
SOFT_LINK_POINTER_FOR_HEADER(PAL, DataDetectorsCore, DDBinderParsecSourceKey, CFStringRef)
SOFT_LINK_POINTER_FOR_HEADER(PAL, DataDetectorsCore, DDBinderHttpURLKey, CFStringRef)
SOFT_LINK_POINTER_FOR_HEADER(PAL, DataDetectorsCore, DDBinderWebURLKey, CFStringRef)
SOFT_LINK_POINTER_FOR_HEADER(PAL, DataDetectorsCore, DDBinderMailURLKey, CFStringRef)
SOFT_LINK_POINTER_FOR_HEADER(PAL, DataDetectorsCore, DDBinderGenericURLKey, CFStringRef)
SOFT_LINK_POINTER_FOR_HEADER(PAL, DataDetectorsCore, DDBinderEmailKey, CFStringRef)
SOFT_LINK_POINTER_FOR_HEADER(PAL, DataDetectorsCore, DDBinderTrackingNumberKey, CFStringRef)
SOFT_LINK_POINTER_FOR_HEADER(PAL, DataDetectorsCore, DDBinderFlightInformationKey, CFStringRef)
SOFT_LINK_POINTER_FOR_HEADER(PAL, DataDetectorsCore, DDBinderSignatureBlockKey, CFStringRef)
SOFT_LINK_CONSTANT_FOR_HEADER(PAL, DataDetectorsCore, DDScannerCopyResultsOptionsForPassiveUse, DDScannerCopyResultsOptions)
SOFT_LINK_FUNCTION_FOR_HEADER(PAL, DataDetectorsCore, DDScannerEnableOptionalSource, void, (DDScannerRef scanner, DDScannerSource source, Boolean enable), (scanner, source, enable))
#endif // PLATFORM(IOS_FAMILY)
SOFT_LINK_FUNCTION_FOR_HEADER(PAL, DataDetectorsCore, DDResultIsPastDate, Boolean, (DDResultRef result, CFDateRef referenceDate, CFTimeZoneRef referenceTimeZone), (result, referenceDate, referenceTimeZone))
SOFT_LINK_FUNCTION_FOR_HEADER(PAL, DataDetectorsCore, DDScannerCreate, DDScannerRef, (DDScannerType type, DDScannerOptions options, CFErrorRef * errorRef), (type, options, errorRef))
SOFT_LINK_FUNCTION_FOR_HEADER(PAL, DataDetectorsCore, DDScannerScanQuery, Boolean, (DDScannerRef scanner, DDScanQueryRef query), (scanner, query))
SOFT_LINK_FUNCTION_FOR_HEADER(PAL, DataDetectorsCore, DDScanQueryCreate, DDScanQueryRef, (CFAllocatorRef allocator), (allocator))
SOFT_LINK_FUNCTION_FOR_HEADER(PAL, DataDetectorsCore, DDScanQueryCreateFromString, DDScanQueryRef, (CFAllocatorRef allocator, CFStringRef string, CFRange range), (allocator, string, range))
SOFT_LINK_FUNCTION_FOR_HEADER(PAL, DataDetectorsCore, DDScannerCopyResultsWithOptions, CFArrayRef, (DDScannerRef scanner, DDScannerCopyResultsOptions options), (scanner, options))
SOFT_LINK_FUNCTION_MAY_FAIL_FOR_HEADER(PAL, DataDetectorsCore, DDResultDisableURLSchemeChecking, void, (), ())
#if HAVE(DDSCANNER_QOS_CONFIGURATION)
SOFT_LINK_FUNCTION_FOR_HEADER(PAL, DataDetectorsCore, DDScannerSetQOS, void, (DDScannerRef scanner, DDQOS qos), (scanner, qos))
#endif
#endif // ENABLE(DATA_DETECTION)
