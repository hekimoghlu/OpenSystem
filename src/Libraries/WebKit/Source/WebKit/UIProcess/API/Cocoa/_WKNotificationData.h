/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 6, 2023.
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

typedef NS_ENUM(NSUInteger, _WKNotificationAlert) {
    _WKNotificationAlertDefault,
    _WKNotificationAlertSilent,
    _WKNotificationAlertEnabled
};

typedef NS_ENUM(NSUInteger, _WKNotificationDirection) {
    _WKNotificationDirectionAuto,
    _WKNotificationDirectionLTR,
    _WKNotificationDirectionRTL
};

WK_CLASS_AVAILABLE(macos(13.3), ios(16.4))
@interface _WKNotificationData : NSObject
- (instancetype)init NS_UNAVAILABLE;

@property (nonatomic, readonly) NSString *title;
@property (nonatomic, readonly) _WKNotificationDirection dir;
@property (nonatomic, readonly) NSString *lang;
@property (nonatomic, readonly) NSString *body;
@property (nonatomic, readonly) NSString *tag;
@property (nonatomic, readonly) _WKNotificationAlert alert;
@property (nonatomic, readonly) NSData *data;

@property (nonatomic, readonly) NSString *origin;
@property (nonatomic, readonly) NSURL *securityOrigin;
@property (nonatomic, readonly) NSURL *serviceWorkerRegistrationURL;

@property (nonatomic, readonly) NSString *identifier;
@property (nonatomic, readonly) NSUUID *uuid;
@property (nonatomic, readonly, copy) NSDictionary *userInfo;

- (NSDictionary *)dictionaryRepresentation;

@end

WK_CLASS_AVAILABLE(macos(15.2), ios(18.2))
@interface _WKMutableNotificationData : _WKNotificationData
- (instancetype)init;

@property (nonatomic, readwrite, copy) NSString *title;
@property (nonatomic, readwrite) _WKNotificationDirection dir;
@property (nonatomic, readwrite, copy) NSString *lang;
@property (nonatomic, readwrite, copy) NSString *body;
@property (nonatomic, readwrite, copy) NSString *tag;
@property (nonatomic, readwrite) _WKNotificationAlert alert;
@property (nonatomic, readwrite, copy) NSData *data;
@property (nonatomic, readwrite, copy) NSURL *securityOrigin;
@property (nonatomic, readwrite, copy) NSURL *serviceWorkerRegistrationURL;
@property (nonatomic, readwrite, copy) NSUUID *uuid;
@end
