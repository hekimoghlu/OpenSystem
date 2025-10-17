/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 3, 2025.
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
#import "WKDataDetectorTypes.h"

#if PLATFORM(IOS_FAMILY) && ENABLE(DATA_DETECTION)

#import <WebCore/DataDetectorType.h>

static inline OptionSet<WebCore::DataDetectorType> fromWKDataDetectorTypes(WKDataDetectorTypes types)
{
    OptionSet<WebCore::DataDetectorType> result;
    if (types & WKDataDetectorTypePhoneNumber)
        result.add(WebCore::DataDetectorType::PhoneNumber);
    if (types & WKDataDetectorTypeLink)
        result.add(WebCore::DataDetectorType::Link);
    if (types & WKDataDetectorTypeAddress)
        result.add(WebCore::DataDetectorType::Address);
    if (types & WKDataDetectorTypeCalendarEvent)
        result.add(WebCore::DataDetectorType::CalendarEvent);
    if (types & WKDataDetectorTypeTrackingNumber)
        result.add(WebCore::DataDetectorType::TrackingNumber);
    if (types & WKDataDetectorTypeFlightNumber)
        result.add(WebCore::DataDetectorType::FlightNumber);
    if (types & WKDataDetectorTypeLookupSuggestion)
        result.add(WebCore::DataDetectorType::LookupSuggestion);

    return result;
}

static inline WKDataDetectorTypes toWKDataDetectorTypes(OptionSet<WebCore::DataDetectorType> types)
{
    WKDataDetectorTypes result { WKDataDetectorTypeNone };
    if (types.contains(WebCore::DataDetectorType::PhoneNumber))
        result = result | WKDataDetectorTypePhoneNumber;
    if (types.contains(WebCore::DataDetectorType::Link))
        result = result | WKDataDetectorTypeLink;
    if (types.contains(WebCore::DataDetectorType::Address))
        result = result | WKDataDetectorTypeAddress;
    if (types.contains(WebCore::DataDetectorType::CalendarEvent))
        result = result | WKDataDetectorTypeCalendarEvent;
    if (types.contains(WebCore::DataDetectorType::TrackingNumber))
        result = result | WKDataDetectorTypeTrackingNumber;
    if (types.contains(WebCore::DataDetectorType::FlightNumber))
        result = result | WKDataDetectorTypeFlightNumber;
    if (types.contains(WebCore::DataDetectorType::LookupSuggestion))
        result = result | WKDataDetectorTypeLookupSuggestion;

    return result;
}


#endif
