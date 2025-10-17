/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 7, 2022.
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
#import "config.h"
#import "_WKNotificationDataInternal.h"

#import <WebCore/NotificationData.h>
#import <WebCore/NotificationDirection.h>
#import <WebCore/SecurityOriginData.h>
#import <WebCore/WebCoreObjCExtras.h>
#import <wtf/cocoa/SpanCocoa.h>
#import <wtf/cocoa/VectorCocoa.h>

static NSString *iconURLKey = @"iconURL";
static NSString *tagKey = @"tag";
static NSString *languageKey = @"language";
static NSString *dataKey = @"data";

@interface _WKNotificationData()
- (instancetype)_init;

@property (nonatomic, readwrite) NSString *title;
@property (nonatomic, readwrite) _WKNotificationDirection dir;
@property (nonatomic, readwrite) NSString *lang;
@property (nonatomic, readwrite) NSString *body;
@property (nonatomic, readwrite) NSString *tag;
@property (nonatomic, readwrite) _WKNotificationAlert alert;
@property (nonatomic, readwrite) NSData *data;
@property (nonatomic, readwrite) NSURL *serviceWorkerRegistrationURL;
@property (nonatomic, readwrite) NSUUID *notificationUUID;

@end

@implementation _WKNotificationData {
@package
    WebCore::NotificationData _coreData;
}

- (instancetype)_init
{
    if (!(self = [super init]))
        return nil;

    return self;
}

- (instancetype)_initWithCoreData:(const WebCore::NotificationData&)coreData
{
    if (!(self = [super init]))
        return nil;

    _coreData = coreData;

    return self;
}

- (const WebCore::NotificationData&)_getCoreData
{
    return _coreData;
}

- (void)setTitle:(NSString *)title
{
    _coreData.title = title;
}

- (NSString *)title
{
    return (NSString *)_coreData.title;
}

- (void)setDir:(_WKNotificationDirection)dir
{
    switch (dir) {
    case _WKNotificationDirectionAuto:
        _coreData.direction = WebCore::NotificationDirection::Auto;
        break;
    case _WKNotificationDirectionLTR:
        _coreData.direction = WebCore::NotificationDirection::Ltr;
        break;
    case _WKNotificationDirectionRTL:
        _coreData.direction = WebCore::NotificationDirection::Rtl;
        break;
    };
}

- (_WKNotificationDirection)dir
{
    switch (_coreData.direction) {
    case WebCore::NotificationDirection::Auto:
        return _WKNotificationDirectionAuto;
    case WebCore::NotificationDirection::Ltr:
        return _WKNotificationDirectionLTR;
    case WebCore::NotificationDirection::Rtl:
        return _WKNotificationDirectionRTL;
    };
}

- (void)setLang:(NSString *)lang
{
    _coreData.language = lang;
}

- (NSString *)lang
{
    return (NSString *)_coreData.language;
}

- (void)setBody:(NSString *)body
{
    _coreData.body = body;
}

- (NSString *)body
{
    return (NSString *)_coreData.body;
}

- (void)setTag:(NSString *)tag
{
    _coreData.tag = tag;
}

- (NSString *)tag
{
    return (NSString *)_coreData.tag;
}

- (void)setAlert:(_WKNotificationAlert)alert
{
    switch (alert) {
    case _WKNotificationAlertDefault:
        _coreData.silent = std::nullopt;
        break;
    case _WKNotificationAlertSilent:
        _coreData.silent = true;
        break;
    case _WKNotificationAlertEnabled:
        _coreData.silent = false;
        break;
    }
}

- (_WKNotificationAlert)alert
{
    if (_coreData.silent == std::nullopt)
        return _WKNotificationAlertDefault;
    return *(_coreData.silent) ? _WKNotificationAlertSilent : _WKNotificationAlertEnabled;
}

- (void)setData:(NSData *)data
{
    _coreData.data = makeVector(data);
}

- (NSData *)data
{
    return toNSData(_coreData.data.span()).autorelease();
}

- (NSString *)origin
{
    return (NSString *)_coreData.originString;
}

- (void)setSecurityOrigin:(NSURL *)securityOrigin
{
    _coreData.originString = WebCore::SecurityOriginData::fromURL(securityOrigin).toString();
}

- (NSURL *)securityOrigin
{
    return (NSURL *)(URL { _coreData.originString });
}

- (void)setServiceWorkerRegistrationURL:(NSURL *)serviceWorkerRegistrationURL
{
    _coreData.serviceWorkerRegistrationURL = serviceWorkerRegistrationURL;
}

- (NSURL *)serviceWorkerRegistrationURL
{
    return (NSURL *)_coreData.serviceWorkerRegistrationURL;
}

- (NSString *)identifier
{
    return (NSString *)_coreData.notificationID.toString();
}

- (void)setUuid:(NSUUID *)uuid
{
    auto wtfUUID = WTF::UUID::fromNSUUID(uuid);
    if (wtfUUID)
        _coreData.notificationID = *wtfUUID;
}

- (NSUUID *)uuid
{
    return (NSUUID *)_coreData.notificationID;
}

- (NSDictionary *)userInfo
{
    return _coreData.dictionaryRepresentation();
}

- (NSDictionary *)dictionaryRepresentation
{
    return [self userInfo];
}

- (void)dealloc
{
    if (WebCoreObjCScheduleDeallocateOnMainRunLoop(_WKNotificationData.class, self))
        return;
    [super dealloc];
}

@end

@implementation _WKMutableNotificationData

@dynamic title;
@dynamic dir;
@dynamic lang;
@dynamic body;
@dynamic tag;
@dynamic alert;
@dynamic data;
@dynamic securityOrigin;
@dynamic serviceWorkerRegistrationURL;
@dynamic uuid;

- (instancetype)init
{
    if (!(self = [super _init]))
        return nil;

    return self;
}

@end
