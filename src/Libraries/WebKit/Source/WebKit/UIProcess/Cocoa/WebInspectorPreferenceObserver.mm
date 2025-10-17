/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 5, 2024.
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
#import "WebInspectorPreferenceObserver.h"

#import "WebInspectorUtilities.h"
#import "WebProcessPool.h"
#import <wtf/RetainPtr.h>
#import <wtf/cocoa/TypeCastsCocoa.h>

@interface WKWebInspectorPreferenceObserver ()
{
@private
    RetainPtr<NSUserDefaults> m_userDefaults;
}
@end

@implementation WKWebInspectorPreferenceObserver

+ (id)sharedInstance
{
    static NeverDestroyed<RetainPtr<WKWebInspectorPreferenceObserver>> instance = adoptNS([[[self class] alloc] init]);
    return instance.get().get();
}

- (instancetype)init
{
    if (!(self = [super init]))
        return nil;

    auto sandboxBrokerBundleIdentifier = WebKit::bundleIdentifierForSandboxBroker();
    m_userDefaults = adoptNS([[NSUserDefaults alloc] initWithSuiteName:bridge_cast(sandboxBrokerBundleIdentifier)]);
    if (!m_userDefaults) {
        WTFLogAlways("Could not init user defaults instance for domain %s.", sandboxBrokerBundleIdentifier);
        return self;
    }
    [m_userDefaults.get() addObserver:self forKeyPath:@"ShowDevelopMenu" options:NSKeyValueObservingOptionNew context:nil];

    
    return self;
}

- (void)observeValueForKeyPath:(NSString *)keyPath ofObject:(id)object change:(NSDictionary<NSKeyValueChangeKey, id> *)change context:(void *)context
{
    RunLoop::main().dispatch([] {
        for (auto& pool : WebKit::WebProcessPool::allProcessPools()) {
            for (size_t i = 0; i < pool->processes().size(); ++i) {
                Ref process = pool->processes()[i];
                process->enableRemoteInspectorIfNeeded();
            }
        }
    });
}

@end
