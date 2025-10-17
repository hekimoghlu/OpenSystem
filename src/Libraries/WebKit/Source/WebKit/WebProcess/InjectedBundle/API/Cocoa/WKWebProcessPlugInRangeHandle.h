/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 21, 2023.
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
#import <WebKit/WKFoundation.h>

#import <Foundation/Foundation.h>
#import <JavaScriptCore/JavaScriptCore.h>
#import <WebKit/WKDataDetectorTypes.h>

@class WKWebProcessPlugInFrame;

WK_CLASS_AVAILABLE(macos(10.12.4), ios(10.3))
@interface WKWebProcessPlugInRangeHandle : NSObject

+ (WKWebProcessPlugInRangeHandle *)rangeHandleWithJSValue:(JSValue *)value inContext:(JSContext *)context;

@property (nonatomic, readonly) WKWebProcessPlugInFrame *frame;
@property (nonatomic, readonly, copy) NSString *text WK_API_AVAILABLE(macos(10.13), ios(11.0));

#if TARGET_OS_IPHONE
- (NSArray *)detectDataWithTypes:(WKDataDetectorTypes)types context:(NSDictionary *)context WK_API_AVAILABLE(ios(11.0));
#endif

@end
