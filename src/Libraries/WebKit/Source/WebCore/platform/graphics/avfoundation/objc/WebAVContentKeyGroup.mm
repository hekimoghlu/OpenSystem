/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 9, 2024.
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
// FIXME (116158267): This file can be removed and its implementation merged directly into
// CDMInstanceSessionFairPlayStreamingAVFObjC once we no logner need to support a configuration
// where the BuiltInCDMKeyGroupingStrategyEnabled preference is off.

#import "config.h"
#import "WebAVContentKeyGroup.h"

#if HAVE(AVCONTENTKEYSESSION)

#import "ContentKeyGroupDataSource.h"
#import "Logging.h"
#import "NotImplemented.h"
#import <wtf/LoggerHelper.h>
#import <wtf/RetainPtr.h>
#import <wtf/Vector.h>
#import <wtf/WeakObjCPtr.h>
#import <wtf/WeakPtr.h>
#import <wtf/text/WTFString.h>

#import <pal/cocoa/AVFoundationSoftLink.h>

NS_ASSUME_NONNULL_BEGIN

#if !RELEASE_LOG_DISABLED
@interface WebAVContentKeyGroup (Logging)
@property (nonatomic, readonly) uint64_t logIdentifier;
@property (nonatomic, readonly) const Logger* loggerPtr;
@property (nonatomic, readonly) WTFLogChannel* logChannel;
@end
#endif

#if HAVE(AVCONTENTKEY_REVOKE)
// FIXME (117803793): Remove staging code once -[AVContentKey revoke] is available in SDKs used by WebKit builders
@interface AVContentKey (Staging_113340014)
- (void)revoke;
@end
#endif

@implementation WebAVContentKeyGroup {
    WeakObjCPtr<AVContentKeySession> _contentKeySession;
    WeakPtr<WebCore::ContentKeyGroupDataSource> _dataSource;
    RetainPtr<NSUUID> _groupIdentifier;
}

- (instancetype)initWithContentKeySession:(AVContentKeySession *)contentKeySession dataSource:(WebCore::ContentKeyGroupDataSource&)dataSource
{
    self = [super init];
    if (!self)
        return nil;

    _contentKeySession = contentKeySession;
    _dataSource = dataSource;
    _groupIdentifier = adoptNS([[NSUUID alloc] init]);

    OBJC_INFO_LOG(OBJC_LOGIDENTIFIER, "groupIdentifier=", _groupIdentifier.get());
    return self;
}

#pragma mark WebAVContentKeyGrouping

- (nullable NSData *)contentProtectionSessionIdentifier
{
    uuid_t uuidBytes = { };
    [_groupIdentifier getUUIDBytes:uuidBytes];
    return [NSData dataWithBytes:uuidBytes length:sizeof(uuidBytes)];
}

- (BOOL)associateContentKeyRequest:(AVContentKeyRequest *)contentKeyRequest
{
    // We assume the data source is tracking unexpected key requests and will include them when
    // ContentKeyGroupDataSource::contentKeyGroupDataSourceKeys() is called in -expire, so there's
    // no need to do any explicit association here.
    OBJC_INFO_LOG(OBJC_LOGIDENTIFIER, "contentKeyRequest=", contentKeyRequest);
    return YES;
}

- (void)expire
{
    if (!_dataSource)
        return;

#if HAVE(AVCONTENTKEY_REVOKE)
    Vector keys = _dataSource->contentKeyGroupDataSourceKeys();
    OBJC_INFO_LOG(OBJC_LOGIDENTIFIER, "keys=", keys.size());

    // FIXME (117803793): Remove staging code once -[AVContentKey revoke] is available in SDKs used by WebKit builders
    if (![PAL::getAVContentKeyClass() instancesRespondToSelector:@selector(revoke)])
        return;

    for (auto& key : keys)
        [key revoke];
#endif
}

- (void)processContentKeyRequestWithIdentifier:(nullable id)identifier initializationData:(nullable NSData *)initializationData options:(nullable NSDictionary<NSString *, id> *)options
{
    OBJC_INFO_LOG(OBJC_LOGIDENTIFIER, identifier, ", initializationData=", initializationData, ", options=", options);
    [_contentKeySession processContentKeyRequestWithIdentifier:identifier initializationData:initializationData options:options];
}

@end

#if !RELEASE_LOG_DISABLED

@implementation WebAVContentKeyGroup (Logging)

- (uint64_t)logIdentifier
{
    return _dataSource ? _dataSource->contentKeyGroupDataSourceLogIdentifier() : 0;
}

- (const Logger*)loggerPtr
{
    return _dataSource ? &_dataSource->contentKeyGroupDataSourceLogger() : nullptr;
}

- (WTFLogChannel*)logChannel
{
    return _dataSource ? &_dataSource->contentKeyGroupDataSourceLogChannel() : nullptr;
}

@end

#endif // !RELEASE_LOG_DISABLED

NS_ASSUME_NONNULL_END

#endif // HAVE(AVCONTENTKEYSESSION)
