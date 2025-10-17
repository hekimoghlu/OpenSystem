/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 3, 2022.
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

#import <Foundation/Foundation.h>

typedef NS_ENUM(NSInteger, WKGroupSessionState) {
    WKGroupSessionStateWaiting = 0,
    WKGroupSessionStateJoined = 1,
    WKGroupSessionStateInvalidated = 2,
};

@class AVPlaybackCoordinator;
@class NSURL;
@class NSUUID;
@class WKGroupSession;
@class WKURLActivity;

NS_ASSUME_NONNULL_BEGIN

@interface WKGroupSessionObserver : NSObject
@property (nonatomic, copy) void (^ _Nullable newSessionCallback)(WKGroupSession *);
- (nonnull instancetype)init;
@end

@interface WKGroupSession : NSObject
@property (nonatomic, readonly, strong) WKURLActivity *activity;
@property (nonatomic, readonly, copy) NSUUID *uuid;
@property (nonatomic, readonly) WKGroupSessionState state;
@property (nonatomic, copy) void (^ _Nullable newActivityCallback)(WKURLActivity *);
@property (nonatomic, copy) void (^ _Nullable stateChangedCallback)(WKGroupSessionState);
- (void)join;
- (void)leave;
- (void)coordinateWithCoordinator:(AVPlaybackCoordinator *)playbackCoordinator;
@end

@interface WKURLActivity : NSObject
@property (nonatomic, copy) NSURL * _Nullable fallbackURL;
@end

NS_ASSUME_NONNULL_END
