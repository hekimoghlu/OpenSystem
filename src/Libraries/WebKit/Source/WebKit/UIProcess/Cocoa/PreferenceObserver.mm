/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 16, 2021.
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
#import "PreferenceObserver.h"

#import "WebProcessPool.h"
#import <pal/spi/cocoa/NSUserDefaultsSPI.h>
#import <wtf/WeakObjCPtr.h>

@interface WKUserDefaults : NSUserDefaults {
@private
    RetainPtr<NSString> m_suiteName;
@public
    WeakObjCPtr<WKPreferenceObserver> m_observer;
}
- (void)findPreferenceChangesAndNotifyForKeys:(NSDictionary<NSString *, id> *)oldValues toValuesForKeys:(NSDictionary<NSString *, id> *)newValues;
@end

@interface WKPreferenceObserver () {
@private
    Vector<RetainPtr<WKUserDefaults>> m_userDefaults;
}
@end

@implementation WKUserDefaults

- (void)findPreferenceChangesAndNotifyForKeys:(NSDictionary<NSString *, id> *)oldValues toValuesForKeys:(NSDictionary<NSString *, id> *)newValues
{
    if (!m_observer)
        return;

    for (NSString *key in oldValues) {
        id oldValue = oldValues[key];
        id newValue = newValues[key];

        if ([oldValue isEqual:newValue])
            continue;

        if (newValue && ![[newValue class] supportsSecureCoding])
            continue;

        NSString *encodedString = nil;

        if (newValue) {
            NSError *e = nil;
            auto data = retainPtr([NSKeyedArchiver archivedDataWithRootObject:newValue requiringSecureCoding:YES error:&e]);
            ASSERT(!e);
            encodedString = [data base64EncodedStringWithOptions:0];
        }

        auto systemValue = adoptCF(CFPreferencesCopyValue((__bridge CFStringRef)key, kCFPreferencesAnyApplication, kCFPreferencesAnyUser, kCFPreferencesAnyHost));
        auto globalValue = adoptCF(CFPreferencesCopyValue((__bridge CFStringRef)key, kCFPreferencesAnyApplication, kCFPreferencesCurrentUser, kCFPreferencesAnyHost));
        auto domainValue = adoptCF(CFPreferencesCopyValue((__bridge CFStringRef)key, (__bridge CFStringRef)m_suiteName.get(), kCFPreferencesCurrentUser, kCFPreferencesAnyHost));

        auto preferenceValuesAreEqual = [] (id a, id b) {
            return a == b || [a isEqual:b];
        };

        if (preferenceValuesAreEqual((__bridge id)systemValue.get(), newValue) || preferenceValuesAreEqual((__bridge id)globalValue.get(), newValue))
            [m_observer preferenceDidChange:nil key:key encodedValue:encodedString];

        if (preferenceValuesAreEqual((__bridge id)domainValue.get(), newValue))
            [m_observer preferenceDidChange:m_suiteName.get() key:key encodedValue:encodedString];
    }
}

- (void)_notifyObserversOfChangeFromValuesForKeys:(NSDictionary<NSString *, id> *)oldValues toValuesForKeys:(NSDictionary<NSString *, id> *)newValues
{
    [super _notifyObserversOfChangeFromValuesForKeys:oldValues toValuesForKeys:newValues];

    if (!isMainRunLoop()) {
        [self findPreferenceChangesAndNotifyForKeys:oldValues toValuesForKeys:newValues];
        return;
    }

    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), [self, protectedSelf = retainPtr(self), oldValues = retainPtr(oldValues), newValues = retainPtr(newValues)] {
        [self findPreferenceChangesAndNotifyForKeys:oldValues.get() toValuesForKeys:newValues.get()];
    });
}

- (void)observeValueForKeyPath:(NSString *)keyPath ofObject:(id)object change:(NSDictionary<NSKeyValueChangeKey, id> *)change context:(void *)context
{
}

- (instancetype)initWithSuiteName:(NSString *)suiteName
{
    if (!(self = [super initWithSuiteName:suiteName]))
        return nil;

    m_suiteName = suiteName;
    m_observer = nil;
    return self;
}
@end

@implementation WKPreferenceObserver

+ (id)sharedInstance
{
    static NeverDestroyed<RetainPtr<WKPreferenceObserver>> instance = adoptNS([[[self class] alloc] init]);
    return instance.get().get();
}

- (instancetype)init
{
    if (!(self = [super init]))
        return nil;

    std::initializer_list<NSString*> domains = {
        @"com.apple.Accessibility",
        @"com.apple.mediaaccessibility",
#if PLATFORM(IOS_FAMILY)
        @"com.apple.AdLib",
        @"com.apple.SpeakSelection",
        @"com.apple.UIKit",
        @"com.apple.WebUI",
        @"com.apple.avfaudio",
        @"com.apple.itunesstored",
        @"com.apple.mediaremote",
        @"com.apple.preferences.sounds",
        @"com.apple.voiceservices",
#else
        @"com.apple.CFNetwork",
        @"com.apple.CoreGraphics",
        @"com.apple.HIToolbox",
        @"com.apple.ServicesMenu.Services",
        @"com.apple.ViewBridge",
        @"com.apple.avfoundation",
        @"com.apple.avfoundation.videoperformancehud",
        @"com.apple.driver.AppleBluetoothMultitouch.mouse",
        @"com.apple.driver.AppleBluetoothMultitouch.trackpad",
        @"com.apple.speech.voice.prefs",
        @"com.apple.universalaccess",
#endif
    };

    for (auto domain : domains) {
        auto userDefaults = adoptNS([[WKUserDefaults alloc] initWithSuiteName:domain]);
        if (!userDefaults) {
            WTFLogAlways("Could not init user defaults instance for domain %s", String(domain).utf8().data());
            continue;
        }
        userDefaults->m_observer = self;
        // Start observing a dummy key in order to make the preference daemon become aware of our NSUserDefaults instance.
        // This is to make sure we receive KVO notifications. We cannot use normal KVO techniques here, since we are looking
        // for _any_ changes in a preference domain. For normal KVO techniques to work, we need to provide the specific
        // key(s) we want to observe, but that set of keys is unknown to us.
        [userDefaults.get() addObserver:userDefaults.get() forKeyPath:@"testkey" options:NSKeyValueObservingOptionNew context:nil];
        m_userDefaults.append(userDefaults);
    }
    return self;
}

- (void)preferenceDidChange:(NSString *)domain key:(NSString *)key encodedValue:(NSString *)encodedValue
{
#if ENABLE(CFPREFS_DIRECT_MODE)
    RunLoop::main().dispatch([domain = retainPtr(domain), key = retainPtr(key), encodedValue = retainPtr(encodedValue)] {
        std::optional<String> encodedValueString;
        if (encodedValue)
            encodedValueString = String(encodedValue.get());
        String domainString = domain.get();
        String keyString = key.get();

#if ENABLE(GPU_PROCESS)
        if (RefPtr gpuProcess = WebKit::GPUProcessProxy::singletonIfCreated())
            gpuProcess->notifyPreferencesChanged(domainString, keyString, encodedValueString);
#endif

        if (RefPtr networkProcess = WebKit::NetworkProcessProxy::defaultNetworkProcess().get())
            networkProcess->notifyPreferencesChanged(domainString, keyString, encodedValueString);

        for (auto& processPool : WebKit::WebProcessPool::allProcessPools())
            processPool->notifyPreferencesChanged(domainString, keyString, encodedValueString);
    });
#endif
}
@end
