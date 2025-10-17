/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 14, 2022.
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
#import "ThermalMitigationNotifier.h"

#if HAVE(APPLE_THERMAL_MITIGATION_SUPPORT)

#import "Logging.h"
#import <Foundation/NSProcessInfo.h>
#import <wtf/MainThread.h>
#import <wtf/TZoneMallocInlines.h>

namespace WebCore {
static bool isThermalMitigationEnabled()
{
    return NSProcessInfo.processInfo.thermalState >= NSProcessInfoThermalStateSerious;
}
}

@interface WebThermalMitigationObserver : NSObject
@property (nonatomic) WeakPtr<WebCore::ThermalMitigationNotifier> notifier;
@property (nonatomic, readonly) BOOL thermalMitigationEnabled;
@end

@implementation WebThermalMitigationObserver

- (WebThermalMitigationObserver *)initWithNotifier:(WebCore::ThermalMitigationNotifier&)notifier
{
    self = [super init];
    if (!self)
        return nil;

    _notifier = &notifier;
    [[NSNotificationCenter defaultCenter] addObserver:self selector:@selector(thermalStateDidChange) name:NSProcessInfoThermalStateDidChangeNotification object:nil];
    return self;
}

- (void)dealloc
{
    [[NSNotificationCenter defaultCenter] removeObserver:self name:NSProcessInfoThermalStateDidChangeNotification object:nil];
    [super dealloc];
}

- (void)thermalStateDidChange
{
    callOnMainThread([self, protectedSelf = RetainPtr<WebThermalMitigationObserver>(self)] {
        if (_notifier)
            notifyThermalMitigationChanged(*_notifier, self.thermalMitigationEnabled);
    });
}

- (BOOL)thermalMitigationEnabled
{
    return WebCore::isThermalMitigationEnabled();
}

@end

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(ThermalMitigationNotifier);

ThermalMitigationNotifier::ThermalMitigationNotifier(ThermalMitigationChangeCallback&& callback)
    : m_observer(adoptNS([[WebThermalMitigationObserver alloc] initWithNotifier:*this]))
    , m_callback(WTFMove(callback))
{
}

ThermalMitigationNotifier::~ThermalMitigationNotifier()
{
    [m_observer setNotifier:nil];
}

bool ThermalMitigationNotifier::thermalMitigationEnabled() const
{
    return [m_observer thermalMitigationEnabled];
}

bool ThermalMitigationNotifier::isThermalMitigationEnabled()
{
    return WebCore::isThermalMitigationEnabled();
}

void ThermalMitigationNotifier::notifyThermalMitigationChanged(bool enabled)
{
    m_callback(enabled);
}

void notifyThermalMitigationChanged(ThermalMitigationNotifier& notifier, bool enabled)
{
    RELEASE_LOG(PerformanceLogging, "Thermal mitigation is enabled: %d", enabled);
    notifier.notifyThermalMitigationChanged(enabled);
}

} // namespace WebCore

#endif // HAVE(APPLE_THERMAL_MITIGATION_SUPPORT)
