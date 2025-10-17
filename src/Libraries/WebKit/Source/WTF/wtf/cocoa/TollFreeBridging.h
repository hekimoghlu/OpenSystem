/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 18, 2023.
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

#ifdef __OBJC__
#import <CoreFoundation/CoreFoundation.h>
#import <Foundation/Foundation.h>
#import <wtf/spi/cocoa/IOSurfaceSPI.h>
#endif

namespace WTF {

template<typename> struct CFTollFreeBridgingTraits;
template<typename> struct NSTollFreeBridgingTraits;

#ifdef __OBJC__

#define WTF_DECLARE_TOLL_FREE_BRIDGING_TRAITS(CFClassName, NSClassName) \
template<> struct CFTollFreeBridgingTraits<CFClassName##Ref> { using BridgedType = NSClassName *; }; \
template<> struct NSTollFreeBridgingTraits<NSClassName> { using BridgedType = CFClassName##Ref; };

WTF_DECLARE_TOLL_FREE_BRIDGING_TRAITS(CFArray, NSArray)
WTF_DECLARE_TOLL_FREE_BRIDGING_TRAITS(CFAttributedString, NSAttributedString)
WTF_DECLARE_TOLL_FREE_BRIDGING_TRAITS(CFCharacterSet, NSCharacterSet)
WTF_DECLARE_TOLL_FREE_BRIDGING_TRAITS(CFData, NSData)
WTF_DECLARE_TOLL_FREE_BRIDGING_TRAITS(CFDate, NSDate)
WTF_DECLARE_TOLL_FREE_BRIDGING_TRAITS(CFDictionary, NSDictionary)
WTF_DECLARE_TOLL_FREE_BRIDGING_TRAITS(CFError, NSError)
WTF_DECLARE_TOLL_FREE_BRIDGING_TRAITS(CFFileSecurity, NSFileSecurity)
WTF_DECLARE_TOLL_FREE_BRIDGING_TRAITS(CFLocale, NSLocale)
WTF_DECLARE_TOLL_FREE_BRIDGING_TRAITS(CFNull, NSNull)
WTF_DECLARE_TOLL_FREE_BRIDGING_TRAITS(CFNumber, NSNumber)
WTF_DECLARE_TOLL_FREE_BRIDGING_TRAITS(CFSet, NSSet)
WTF_DECLARE_TOLL_FREE_BRIDGING_TRAITS(CFString, NSString)
WTF_DECLARE_TOLL_FREE_BRIDGING_TRAITS(CFTimeZone, NSTimeZone)
WTF_DECLARE_TOLL_FREE_BRIDGING_TRAITS(CFURL, NSURL)
WTF_DECLARE_TOLL_FREE_BRIDGING_TRAITS(IOSurface, ::IOSurface)

template<> struct CFTollFreeBridgingTraits<CFBooleanRef> { using BridgedType = NSNumber *; };

WTF_DECLARE_TOLL_FREE_BRIDGING_TRAITS(CFMutableArray, NSMutableArray)
WTF_DECLARE_TOLL_FREE_BRIDGING_TRAITS(CFMutableAttributedString, NSMutableAttributedString)
WTF_DECLARE_TOLL_FREE_BRIDGING_TRAITS(CFMutableData, NSMutableData)
WTF_DECLARE_TOLL_FREE_BRIDGING_TRAITS(CFMutableDictionary, NSMutableDictionary)
WTF_DECLARE_TOLL_FREE_BRIDGING_TRAITS(CFMutableSet, NSMutableSet)
WTF_DECLARE_TOLL_FREE_BRIDGING_TRAITS(CFMutableString, NSMutableString)

#undef WTF_DECLARE_TOLL_FREE_BRIDGING_TRAITS

#endif // defined(__OBJC__)

} // namespace WTF
