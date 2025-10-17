/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 25, 2023.
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
#import "NetworkStateNotifier.h"

#if PLATFORM(IOS_FAMILY)

#import "DeprecatedGlobalSettings.h"
#import "WebCoreThreadRun.h"
#import <wtf/BlockPtr.h>

#if USE(APPLE_INTERNAL_SDK)
#import <AppSupport/CPNetworkObserver.h>
#else
@interface CPNetworkObserver : NSObject
+ (CPNetworkObserver *)sharedNetworkObserver;
- (void)addNetworkReachableObserver:(id)observer selector:(SEL)selector;
- (BOOL)isNetworkReachable;
@end
#endif

@interface WebNetworkStateObserver : NSObject {
    BlockPtr<void()> block;
}
- (id)initWithBlock:(void (^)())block;
@end

@implementation WebNetworkStateObserver

- (id)initWithBlock:(void (^)())observerBlock
{
    if (!(self = [super init]))
        return nil;
    [[CPNetworkObserver sharedNetworkObserver] addNetworkReachableObserver:self selector:@selector(networkStateChanged:)];
    block = makeBlockPtr(observerBlock);
    return self;
}

- (void)networkStateChanged:(NSNotification *)unusedNotification
{
    UNUSED_PARAM(unusedNotification);
    block();
}

@end

namespace WebCore {

void NetworkStateNotifier::updateStateWithoutNotifying()
{
    m_isOnLine = [[CPNetworkObserver sharedNetworkObserver] isNetworkReachable];
}

void NetworkStateNotifier::startObserving()
{
    if (DeprecatedGlobalSettings::shouldOptOutOfNetworkStateObservation())
        return;
    m_observer = adoptNS([[WebNetworkStateObserver alloc] initWithBlock:^ {
        callOnMainThread([] {
            NetworkStateNotifier::singleton().updateStateSoon();
        });
    }]);
}

} // namespace WebCore

#endif // PLATFORM(IOS_FAMILY)
