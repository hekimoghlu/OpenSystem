/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 27, 2021.
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

@class _WKAutomationSessionConfiguration;
@protocol _WKAutomationSessionDelegate;

NS_ASSUME_NONNULL_BEGIN

WK_CLASS_AVAILABLE(macos(10.12), ios(10.0))
@interface _WKAutomationSession : NSObject

@property (nonatomic, copy) NSString *sessionIdentifier;
@property (nonatomic, readonly, copy) _WKAutomationSessionConfiguration *configuration;

@property (nonatomic, weak) id <_WKAutomationSessionDelegate> delegate;
@property (nonatomic, readonly, getter=isPaired) BOOL paired;
@property (nonatomic, readonly, getter=isPendingTermination) BOOL pendingTermination WK_API_AVAILABLE(macos(13.0), ios(16.0));

@property (nonatomic, readonly, getter=isSimulatingUserInteraction) BOOL simulatingUserInteraction WK_API_AVAILABLE(macos(10.13.4), ios(11.3));

- (instancetype)initWithConfiguration:(_WKAutomationSessionConfiguration *)configuration NS_DESIGNATED_INITIALIZER;

- (void)terminate WK_API_AVAILABLE(macos(10.14), ios(12.0));

#if !TARGET_OS_IPHONE
- (BOOL)wasEventSynthesizedForAutomation:(NSEvent *)event;
- (void)markEventAsSynthesizedForAutomation:(NSEvent *)event;
#endif

@end

NS_ASSUME_NONNULL_END
