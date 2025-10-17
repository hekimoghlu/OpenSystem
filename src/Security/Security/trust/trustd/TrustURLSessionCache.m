/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 6, 2025.
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
#import <Foundation/Foundation.h>
#import <CFNetwork/CFNSURLConnection.h>
#include <Security/SecInternalReleasePriv.h>
#include <utilities/debugging.h>

#include "trust/trustd/TrustURLSessionDelegate.h"
#include "trust/trustd/TrustURLSessionCache.h"

#define MAX_CACHED_SESSIONS 20
const NSString *TrustdUserAgent = @"com.apple.trustd/3.0";

NSTimeInterval TrustURLSessionGetResourceTimeout(void) {
    return (NSTimeInterval)3.0;
}


@interface TrustURLSessionCache()
@property TrustURLSessionDelegate *delegate;
@property NSMutableDictionary <NSData *, NSURLSession *>* _clientSessionMap;
@property NSMutableArray <NSData *>* _clientLRUList;
@property _NSHSTSStorage * _sharedHSTSCache;
@end

@implementation TrustURLSessionCache

- (instancetype)initWithDelegate:(TrustURLSessionDelegate *)delegate
{
    if (self = [super init]) {
        self.delegate = delegate;
        self._clientSessionMap = [NSMutableDictionary dictionaryWithCapacity:MAX_CACHED_SESSIONS];
        self._clientLRUList = [NSMutableArray arrayWithCapacity:(MAX_CACHED_SESSIONS + 1)];
        self._sharedHSTSCache = [[_NSHSTSStorage alloc] initInMemoryStore];
    }
    return self;
}

- (NSURLSession *)createSessionForAuditToken:(NSData *)auditToken
{
    NSURLSessionConfiguration *config = [NSURLSessionConfiguration ephemeralSessionConfiguration];
    config._hstsStorage = self._sharedHSTSCache; // use shared ephemeral HSTS cache
    config.HTTPCookieStorage = nil; // no cookies
    config.URLCache = nil; // no resource caching
    config.HTTPAdditionalHeaders = @{@"User-Agent" : TrustdUserAgent};
    config._sourceApplicationAuditTokenData = auditToken;
    config._sourceApplicationSecondaryIdentifier = @"com.apple.trustd.TrustURLSession"; // Must match NetworkServiceProxy definition

    NSOperationQueue *queue = [[NSOperationQueue alloc] init];
    queue.underlyingQueue = self.delegate.queue;

    NSURLSession *session = [NSURLSession sessionWithConfiguration:config delegate:self.delegate delegateQueue:queue];
    return session;
}

- (NSURLSession *)sessionForAuditToken:(NSData *)auditToken
{
    @synchronized (self._clientLRUList) {
        NSURLSession *result = [self._clientSessionMap objectForKey:auditToken];
        if (result) {
            /* insert the client to the front of the LRU list */
            [self._clientLRUList removeObject:auditToken];
            [self._clientLRUList insertObject:auditToken atIndex:0];
            secdebug("http", "re-using session for %@", auditToken);
            return result;
        }
        /* Cache miss: create new session */
        result = [self createSessionForAuditToken:auditToken];
        [self._clientLRUList insertObject:auditToken atIndex:0];
        [self._clientSessionMap setObject:result forKey:auditToken];
        secdebug("http", "creating session for %@", auditToken);
        if (self._clientLRUList.count > MAX_CACHED_SESSIONS) {
            /* close the excess NSURLSession and remove it from our cache */
            NSData *removeToken = [self._clientLRUList objectAtIndex:(self._clientLRUList.count - 1)];
            NSURLSession *removeSession = [self._clientSessionMap objectForKey:removeToken];
            [removeSession finishTasksAndInvalidate];
            [self._clientSessionMap removeObjectForKey:removeToken];
            [self._clientLRUList removeLastObject];
            secdebug("http", "removing session for %@", removeToken);
        }
        return result;
    }
}

@end
