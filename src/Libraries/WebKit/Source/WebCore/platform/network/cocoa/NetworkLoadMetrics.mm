/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 30, 2025.
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
#import "NetworkLoadMetrics.h"

#import "ResourceHandle.h"
#import <pal/spi/cocoa/NSURLConnectionSPI.h>

namespace WebCore {

static MonotonicTime dateToMonotonicTime(NSDate *date)
{
    if (auto interval = date.timeIntervalSince1970)
        return WallTime::fromRawSeconds(interval).approximateMonotonicTime();
    return { };
}

static Box<NetworkLoadMetrics> packageTimingData(MonotonicTime redirectStart, NSDate *fetchStart, NSDate *domainLookupStart, NSDate *domainLookupEnd, NSDate *connectStart, NSDate *secureConnectionStart, NSDate *connectEnd, NSDate *requestStart, NSDate *responseStart, bool reusedTLSConnection, NSString *protocol, uint16_t redirectCount, bool failsTAOCheck, bool hasCrossOriginRedirect)
{

    auto timing = Box<NetworkLoadMetrics>::create();

    timing->redirectStart = redirectStart;
    timing->fetchStart = dateToMonotonicTime(fetchStart);
    timing->domainLookupStart = dateToMonotonicTime(domainLookupStart);
    timing->domainLookupEnd = dateToMonotonicTime(domainLookupEnd);
    timing->connectStart = dateToMonotonicTime(connectStart);
    if (reusedTLSConnection && [protocol isEqualToString:@"https"])
        timing->secureConnectionStart = reusedTLSConnectionSentinel;
    else
        timing->secureConnectionStart = dateToMonotonicTime(secureConnectionStart);
    timing->connectEnd = dateToMonotonicTime(connectEnd);
    timing->requestStart = dateToMonotonicTime(requestStart);
    // Sometimes, likely because of <rdar://90997689>, responseStart is before requestStart. If this happens, use the later of the two.
    timing->responseStart = std::max(timing->requestStart, dateToMonotonicTime(responseStart));
    timing->redirectCount = redirectCount;
    timing->failsTAOCheck = failsTAOCheck;
    timing->hasCrossOriginRedirect = hasCrossOriginRedirect;

    // NOTE: responseEnd is not populated in this code path.

    return timing;
}

Box<NetworkLoadMetrics> copyTimingData(NSURLSessionTaskMetrics *incompleteMetrics, const NetworkLoadMetrics& metricsFromTask)
{
    NSArray<NSURLSessionTaskTransactionMetrics *> *transactionMetrics = incompleteMetrics.transactionMetrics;
    NSURLSessionTaskTransactionMetrics *metrics = transactionMetrics.lastObject;
    return packageTimingData(
        dateToMonotonicTime(transactionMetrics.firstObject.fetchStartDate),
        metrics.fetchStartDate,
        metrics.domainLookupStartDate,
        metrics.domainLookupEndDate,
        metrics.connectStartDate,
        metrics.secureConnectionStartDate,
        metrics.connectEndDate,
        metrics.requestStartDate,
        metrics.responseStartDate,
        metrics.reusedConnection,
        metrics.response.URL.scheme,
        incompleteMetrics.redirectCount,
        metricsFromTask.failsTAOCheck,
        metricsFromTask.hasCrossOriginRedirect
    );
}

Box<NetworkLoadMetrics> copyTimingData(NSURLConnection *connection, const ResourceHandle& handle)
{
    NSDictionary *timingData = [connection _timingData];

    auto timingValue = [&](NSString *key) -> RetainPtr<NSDate> {
        if (NSNumber *number = [timingData objectForKey:key]) {
            if (double doubleValue = number.doubleValue)
                return adoptNS([[NSDate alloc] initWithTimeIntervalSinceReferenceDate:doubleValue]);
        }
        return { };
    };

    auto data = packageTimingData(
        handle.startTimeBeforeRedirects(),
        timingValue(@"_kCFNTimingDataFetchStart").get(),
        timingValue(@"_kCFNTimingDataDomainLookupStart").get(),
        timingValue(@"_kCFNTimingDataDomainLookupEnd").get(),
        timingValue(@"_kCFNTimingDataConnectStart").get(),
        timingValue(@"_kCFNTimingDataSecureConnectionStart").get(),
        timingValue(@"_kCFNTimingDataConnectEnd").get(),
        timingValue(@"_kCFNTimingDataRequestStart").get(),
        timingValue(@"_kCFNTimingDataResponseStart").get(),
        timingValue(@"_kCFNTimingDataConnectionReused").get(),
        connection.currentRequest.URL.scheme,
        handle.redirectCount(),
        handle.failsTAOCheck(),
        handle.hasCrossOriginRedirect()
    );

    if (!data->fetchStart)
        data->fetchStart = data->redirectStart;

    return data;
}
    
}
