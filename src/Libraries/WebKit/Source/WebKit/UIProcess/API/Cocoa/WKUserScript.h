/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 12, 2024.
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

NS_ASSUME_NONNULL_BEGIN

@class WKContentWorld;

/*! @enum WKUserScriptInjectionTime
 @abstract when a user script should be injected into a webpage.
 @constant WKUserScriptInjectionTimeAtDocumentStart    Inject the script after the document element has been created, but before any other content has been loaded.
 @constant WKUserScriptInjectionTimeAtDocumentEnd      Inject the script after the document has finished loading, but before any subresources may have finished loading.
 */
typedef NS_ENUM(NSInteger, WKUserScriptInjectionTime) {
    WKUserScriptInjectionTimeAtDocumentStart,
    WKUserScriptInjectionTimeAtDocumentEnd
} WK_API_AVAILABLE(macos(10.10), ios(8.0));

/*! A @link WKUserScript @/link object represents a script that can be injected into webpages.
 */
WK_SWIFT_UI_ACTOR
WK_CLASS_AVAILABLE(macos(10.10), ios(8.0))
@interface WKUserScript : NSObject <NSCopying>

/* @abstract The script source code. */
@property (nonatomic, readonly, copy) NSString *source;

/* @abstract When the script should be injected. */
@property (nonatomic, readonly) WKUserScriptInjectionTime injectionTime;

/* @abstract Whether the script should be injected into all frames or just the main frame. */
@property (nonatomic, readonly, getter=isForMainFrameOnly) BOOL forMainFrameOnly;

/*! @abstract Returns an initialized user script that can be added to a @link WKUserContentController @/link.
 @param source The script source.
 @param injectionTime When the script should be injected.
 @param forMainFrameOnly Whether the script should be injected into all frames or just the main frame.
 @discussion Calling this method is the same as calling `initWithSource:injectionTime:forMainFrameOnly:inContentWorld:` with a `contentWorld` value of `WKContentWorld.pageWorld`
 */
- (instancetype)initWithSource:(NSString *)source injectionTime:(WKUserScriptInjectionTime)injectionTime forMainFrameOnly:(BOOL)forMainFrameOnly;

/*! @abstract Returns an initialized user script that can be added to a @link WKUserContentController @/link.
 @param source The script source.
 @param injectionTime When the script should be injected.
 @param forMainFrameOnly Whether the script should be injected into all frames or just the main frame.
 @param contentWorld The WKContentWorld in which to inject the script.
 */
- (instancetype)initWithSource:(NSString *)source injectionTime:(WKUserScriptInjectionTime)injectionTime forMainFrameOnly:(BOOL)forMainFrameOnly inContentWorld:(WKContentWorld *)contentWorld WK_API_AVAILABLE(macos(11.0), ios(14.0));

@end

NS_ASSUME_NONNULL_END
