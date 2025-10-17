/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 9, 2022.
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
#import "WebSecurityOriginInternal.h"

#import "WebDatabaseQuotaManager.h"
#import "WebQuotaManager.h"
#import <WebCore/DatabaseTracker.h>
#import <WebCore/SecurityOrigin.h>
#import <WebCore/SecurityOriginData.h>
#import <wtf/URL.h>

using namespace WebCore;

@implementation WebSecurityOrigin

+ (id)webSecurityOriginFromDatabaseIdentifier:(NSString *)databaseIdentifier
{
    WTF::initializeMainThread();

    auto origin = SecurityOriginData::fromDatabaseIdentifier(String { databaseIdentifier });
    if (!origin)
        return nil;

    return adoptNS([[WebSecurityOrigin alloc] _initWithWebCoreSecurityOrigin:origin->securityOrigin().ptr()]).autorelease();
}

- (id)initWithURL:(NSURL *)url
{
    WTF::initializeMainThread();

    self = [super init];
    if (!self)
        return nil;

    _private = reinterpret_cast<WebSecurityOriginPrivate *>(&SecurityOrigin::create(URL([url absoluteURL])).leakRef());
    return self;
}

- (NSString *)protocol
{
    return reinterpret_cast<SecurityOrigin*>(_private)->protocol();
}

- (NSString *)host
{
    return reinterpret_cast<SecurityOrigin*>(_private)->host();
}

- (NSString *)databaseIdentifier
{
    return reinterpret_cast<SecurityOrigin*>(_private)->data().databaseIdentifier();
}

#if PLATFORM(IOS_FAMILY)
- (NSString *)toString
{
    return reinterpret_cast<SecurityOrigin*>(_private)->toString();
}
#endif

- (NSString *)stringValue
{
    return reinterpret_cast<SecurityOrigin*>(_private)->toString();
}

- (unsigned short)port
{
    return reinterpret_cast<SecurityOrigin*>(_private)->port().value_or(0);
}

// FIXME: Overriding isEqual: without overriding hash will cause trouble if this ever goes into an NSSet or is the key in an NSDictionary,
// since two equal objects could have different hashes.
- (BOOL)isEqual:(id)anObject
{
    if (![anObject isMemberOfClass:[WebSecurityOrigin class]])
        return NO;
    
    return [self _core]->equal(*[anObject _core]);
}

- (void)dealloc
{
    if (_private)
        reinterpret_cast<SecurityOrigin*>(_private)->deref();
    if (_databaseQuotaManager)
        [(NSObject *)_databaseQuotaManager release];
    [super dealloc];
}

@end

@implementation WebSecurityOrigin (WebInternal)

- (id)_initWithWebCoreSecurityOrigin:(SecurityOrigin*)origin
{
    ASSERT(origin);
    self = [super init];
    if (!self)
        return nil;

    origin->ref();
    _private = reinterpret_cast<WebSecurityOriginPrivate *>(origin);

    return self;
}

- (id)_initWithString:(NSString *)originString
{
    auto origin = SecurityOrigin::createFromString(originString);
    return adoptNS([[WebSecurityOrigin alloc] _initWithWebCoreSecurityOrigin:origin.ptr()]).autorelease();
}

- (SecurityOrigin *)_core
{
    return reinterpret_cast<SecurityOrigin*>(_private);
}

@end


// MARK: -
// MARK: WebQuotaManagers

@implementation WebSecurityOrigin (WebQuotaManagers)

- (id<WebQuotaManager>)databaseQuotaManager
{
    if (!_databaseQuotaManager)
        _databaseQuotaManager = [[WebDatabaseQuotaManager alloc] initWithOrigin:self];
    return _databaseQuotaManager;
}

@end


// MARK: -
// MARK: Deprecated

// FIXME: The following methods are deprecated and should removed later.
// Clients should instead get a WebQuotaManager, and query / set the quota via the Manager.

@implementation WebSecurityOrigin (Deprecated)

- (unsigned long long)usage
{
    return DatabaseTracker::singleton().usage(reinterpret_cast<SecurityOrigin*>(_private)->data());
}

- (unsigned long long)quota
{
    return DatabaseTracker::singleton().quota(reinterpret_cast<SecurityOrigin*>(_private)->data());
}

- (void)setQuota:(unsigned long long)quota
{
    DatabaseTracker::singleton().setQuota(reinterpret_cast<SecurityOrigin*>(_private)->data(), quota);
}

@end
