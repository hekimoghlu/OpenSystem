/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 15, 2021.
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
#import "SecurityAnalyticsReporterRTC.h"
#import "SecurityAnalyticsConstants.h"
#import <SoftLinking/SoftLinking.h>

#if __has_include(<AAAFoundation/AAAFoundation.h>)
SOFT_LINK_OPTIONAL_FRAMEWORK(PrivateFrameworks, AAAFoundation);
SOFT_LINK_CLASS(AAAFoundation, AAFAnalyticsTransportRTC);
SOFT_LINK_CLASS(AAAFoundation, AAFAnalyticsReporter);
#endif

@implementation SecurityAnalyticsReporterRTC

#if __has_include(<AAAFoundation/AAAFoundation.h>)
+ (AAFAnalyticsReporter *)rtcAnalyticsReporter {
    static AAFAnalyticsReporter *rtcReporter = nil;
    static dispatch_once_t onceToken;

    dispatch_once(&onceToken, ^{
        AAFAnalyticsTransportRTC *transport = [getAAFAnalyticsTransportRTCClass() analyticsTransportRTCWithClientType:kSecurityRTCClientType
                                                                                                       clientBundleId:kSecurityRTCClientBundleIdentifier
                                                                                                           clientName:kSecurityRTCClientNameDNU];
        rtcReporter = [getAAFAnalyticsReporterClass() analyticsReporterWithTransport:transport];
    });
    return rtcReporter;
}
#endif

+ (void)sendMetricWithEvent:(AAFAnalyticsEventSecurity*)eventS success:(BOOL)success error:(NSError* _Nullable)error
{
#if __has_include(<AAAFoundation/AAAFoundation.h>)
    
    if ([eventS permittedToSendMetrics] == NO) {
        return;
    }

    dispatch_sync(eventS.queue, ^{
        AAFAnalyticsEvent* event = (AAFAnalyticsEvent*)[eventS getEvent];
        event[kSecurityRTCFieldDidSucceed] = @(success);
        [event populateUnderlyingErrorsStartingWithRootError:error];
        [[SecurityAnalyticsReporterRTC rtcAnalyticsReporter] sendEvent:event];
    });
#endif
}

@end
