/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 9, 2022.
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
#import "WebBackgroundTaskController.h"

#if PLATFORM(IOS_FAMILY)

@implementation WebBackgroundTaskController

+ (WebBackgroundTaskController *)sharedController
{
    static NeverDestroyed<RetainPtr<WebBackgroundTaskController>> sharedController = adoptNS([[self alloc] init]);
    return sharedController.get().get();
}

- (void)dealloc
{
    [_backgroundTaskStartBlock release];
    [_backgroundTaskEndBlock release];
    [super dealloc];
}

- (NSUInteger)startBackgroundTaskWithExpirationHandler:(void (^)())handler
{
    if (!_backgroundTaskStartBlock)
        return _invalidBackgroundTaskIdentifier;
    return _backgroundTaskStartBlock(handler);
}

- (void)endBackgroundTaskWithIdentifier:(NSUInteger)identifier
{
    if (!_backgroundTaskEndBlock)
        return;
    _backgroundTaskEndBlock(identifier);
}

@end

#endif
