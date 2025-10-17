/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 19, 2025.
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
#import "WebDatabaseQuotaManager.h"

#import "WebSecurityOriginInternal.h"
#import <WebCore/DatabaseTracker.h>
#import <WebCore/SecurityOriginData.h>

using namespace WebCore;

@implementation WebDatabaseQuotaManager

- (id)initWithOrigin:(WebSecurityOrigin *)origin
{
    if (!origin) {
        [self release];
        return nil;
    }

    self = [super init];
    if (!self)
        return nil;

    _origin = origin;
    return self;
}

- (WebSecurityOrigin *)origin
{
    return _origin;
}

- (unsigned long long)usage
{
    return DatabaseTracker::singleton().usage([_origin _core]->data());
}

- (unsigned long long)quota
{
    return DatabaseTracker::singleton().quota([_origin _core]->data());
}

// If the quota is set to a value lower than the current usage, that quota will
// "stick" but no data will be purged to meet the new quota. This will simply
// prevent new data from being added to databases in that origin.
- (void)setQuota:(unsigned long long)quota
{
    DatabaseTracker::singleton().setQuota([_origin _core]->data(), quota);
}

@end
