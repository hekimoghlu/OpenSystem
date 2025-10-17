/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 9, 2023.
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

struct WebSecurityOriginPrivate;

@protocol WebQuotaManager;

@interface WebSecurityOrigin : NSObject {
@private
    struct WebSecurityOriginPrivate *_private;
    id<WebQuotaManager> _databaseQuotaManager;
}

+ (id)webSecurityOriginFromDatabaseIdentifier:(NSString *)databaseIdentifier;

- (id)initWithURL:(NSURL *)url;

- (NSString *)protocol;
- (NSString *)host;

- (NSString *)databaseIdentifier;
#if TARGET_OS_IPHONE
- (NSString *)toString;
#endif
- (NSString *)stringValue;

// Returns zero if the port is the default port for the protocol, non-zero otherwise.
- (unsigned short)port;

@end

@interface WebSecurityOrigin (WebQuotaManagers)
- (id<WebQuotaManager>)databaseQuotaManager;
@end

// FIXME: The following methods are deprecated and should removed later.
// Clients should instead get a WebQuotaManager, and query / set the quota via the Manager.
@interface WebSecurityOrigin (Deprecated)
- (unsigned long long)usage;
- (unsigned long long)quota;
- (void)setQuota:(unsigned long long)quota;
@end
