/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 7, 2025.
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
#if __OBJC2__

#import "SFAnalyticsMultiSampler+Internal.h"
#import "SFAnalytics+Internal.h"
#import "SFAnalyticsDefines.h"
#import "utilities/debugging.h"
#include <notify.h>
#include <dispatch/dispatch.h>

@implementation SFAnalyticsMultiSampler {
    NSTimeInterval _samplingInterval;
    dispatch_source_t _timer;
    NSString* _name;
    MultiSamplerDictionary (^_block)(void);
    int _notificationToken;
    Class _clientClass;
    BOOL _oncePerReport;
    BOOL _activeTimer;
}

@synthesize name = _name;
@synthesize samplingInterval = _samplingInterval;
@synthesize oncePerReport = _oncePerReport;

- (instancetype)initWithName:(NSString*)name interval:(NSTimeInterval)interval block:(MultiSamplerDictionary (^)(void))block clientClass:(Class)clientClass
{
    if (self = [super init]) {
        if (![clientClass isSubclassOfClass:[SFAnalytics class]]) {
            secerror("SFAnalyticsSampler created without valid client class (%@)", clientClass);
            return nil;
        }
        
        if (!name || (interval < 1.0f && interval != SFAnalyticsSamplerIntervalOncePerReport) || !block) {
            secerror("SFAnalyticsSampler created without proper data");
            return nil;
        }
        
        _clientClass = clientClass;
        _block = block;
        _name = name;
        _samplingInterval = interval;
        [self newTimer];
    }
    return self;
}

- (void)newTimer
{
    if (_activeTimer) {
        [self pauseSampling];
    }

    _oncePerReport = (_samplingInterval == SFAnalyticsSamplerIntervalOncePerReport);
    if (_oncePerReport) {
        [self setupOnceTimer];
    } else {
        [self setupPeriodicTimer];
    }
}

- (void)setupOnceTimer
{
    __weak __typeof(self) weakSelf = self;
    notify_register_dispatch(SFAnalyticsFireSamplersNotification, &_notificationToken, dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^(int token) {
        __strong __typeof(self) strongSelf = weakSelf;
        if (!strongSelf) {
            secnotice("SFAnalyticsSampler", "sampler went away before we could run its once-per-report block");
            notify_cancel(token);
            return;
        }

        MultiSamplerDictionary data = strongSelf->_block();
        [data enumerateKeysAndObjectsUsingBlock:^(NSString * _Nonnull key, NSNumber * _Nonnull obj, BOOL * _Nonnull stop) {
            [[strongSelf->_clientClass logger] logMetric:obj withName:key oncePerReport:strongSelf->_oncePerReport];
        }];
    });
    _activeTimer = YES;
}

- (void)setupPeriodicTimer
{
    _timer = dispatch_source_create(DISPATCH_SOURCE_TYPE_TIMER, 0, 0, dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0));
    dispatch_source_set_timer(_timer, dispatch_walltime(0, _samplingInterval * NSEC_PER_SEC), _samplingInterval * NSEC_PER_SEC, _samplingInterval * NSEC_PER_SEC / 50.0);    // give 2% leeway on timer

    __weak __typeof(self) weakSelf = self;
    dispatch_source_set_event_handler(_timer, ^{
        __strong __typeof(self) strongSelf = weakSelf;
        if (!strongSelf) {
            secnotice("SFAnalyticsSampler", "sampler went away before we could run its once-per-report block");
            return;
        }

        MultiSamplerDictionary data = strongSelf->_block();
        [data enumerateKeysAndObjectsUsingBlock:^(NSString * _Nonnull key, NSNumber * _Nonnull obj, BOOL * _Nonnull stop) {
            [[strongSelf->_clientClass logger] logMetric:obj withName:key oncePerReport:strongSelf->_oncePerReport];
        }];
    });
    dispatch_resume(_timer);
    
    _activeTimer = YES;
}

- (void)setSamplingInterval:(NSTimeInterval)interval
{
    if (interval < 1.0f && !(interval == SFAnalyticsSamplerIntervalOncePerReport)) {
        secerror("SFAnalyticsSampler: interval %f is not supported", interval);
        return;
    }

    _samplingInterval = interval;
    [self newTimer];
}

- (NSTimeInterval)samplingInterval {
    return _samplingInterval;
}

- (MultiSamplerDictionary)sampleNow
{
    MultiSamplerDictionary data = _block();
    [data enumerateKeysAndObjectsUsingBlock:^(NSString * _Nonnull key, NSNumber * _Nonnull obj, BOOL * _Nonnull stop) {
        [[self->_clientClass logger] logMetric:obj withName:key oncePerReport:self->_oncePerReport];
    }];
    return data;
}

- (void)pauseSampling
{
    if (!_activeTimer) {
        return;
    }

    if (_oncePerReport) {
        notify_cancel(_notificationToken);
        _notificationToken = 0;
    } else {
        dispatch_source_cancel(_timer);
    }
    _activeTimer = NO;
}

- (void)resumeSampling
{
    [self newTimer];
}

- (void)dealloc
{
    [self pauseSampling];
}

@end

#endif
