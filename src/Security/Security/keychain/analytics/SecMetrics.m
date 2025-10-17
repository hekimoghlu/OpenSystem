/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 15, 2023.
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
#import <os/transaction_private.h>
#import <Foundation/Foundation.h>
#import <CFNetwork/CFNetworkPriv.h>
#import <Accounts/Accounts.h>
#import <Accounts/Accounts_Private.h>
#import <AppleAccount/ACAccount+AppleAccount.h>
#import <AppleAccount/ACAccountStore+AppleAccount.h>
#import <zlib.h>

#import "keychain/analytics/SecMetrics.h"
#import "keychain/analytics/SecEventMetric.h"
#import "keychain/analytics/SecEventMetric_private.h"
#import "keychain/analytics/SecC2DeviceInfo.h"

#import "keychain/analytics/C2Metric/SECC2MPMetric.h"
#import "keychain/analytics/C2Metric/SECC2MPGenericEvent.h"
#import "keychain/analytics/C2Metric/SECC2MPGenericEventMetric.h"
#import "keychain/analytics/C2Metric/SECC2MPGenericEventMetricValue.h"
#import "keychain/analytics/C2Metric/SECC2MPDeviceInfo.h"
#import <utilities/SecCoreAnalytics.h>

#import <utilities/simulatecrash_assert.h>



@interface SecMetrics () <NSURLSessionDelegate>
@property (strong) NSMutableDictionary<NSNumber *, SecEventMetric *>  *taskMap;
@property (strong) NSURLSession *URLSession;
@property (strong) os_transaction_t transaction;
@property (assign) long lostEvents;
@end


static NSString *securtitydPushTopic = @"com.apple.private.alloy.keychain.metrics";

@implementation SecMetrics

+ (NSURL *)c2MetricsEndpoint {
    ACAccountStore *store = [ACAccountStore defaultStore];
    ACAccount* primaryAccount = [store aa_primaryAppleAccount];
    if(!primaryAccount) {
        return nil;
    }
    NSString *urlString = [primaryAccount propertiesForDataclass:ACAccountDataclassCKMetricsService][@"url"];
    if (urlString == NULL) {
        return nil;
    }

    NSURL *url = [[[NSURL alloc] initWithString:urlString] URLByAppendingPathComponent:@"c2"];

    if (url) {
        static dispatch_once_t onceToken;
        dispatch_once(&onceToken, ^{
            os_log(OS_LOG_DEFAULT, "metrics URL is: %@", url);
        });
    }

    return url;
}

+ (SecMetrics *)managerObject {
    static SecMetrics *manager;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        manager = [[SecMetrics alloc] init];
    });
    return manager;
}

- (instancetype)init
{
    if ((self = [super init]) == NULL) {
        return self;
    }

    NSURLSessionConfiguration *configuration = [NSURLSessionConfiguration ephemeralSessionConfiguration];

    self.URLSession = [NSURLSession sessionWithConfiguration:configuration delegate:self delegateQueue:NULL];
    self.taskMap = [NSMutableDictionary dictionary];
    return self;
}

- (void)submitEvent:(SecEventMetric *)metric
{
    [self sendEvent:metric pushTopic:securtitydPushTopic];
}


- (void)sendEvent:(SecEventMetric *)event pushTopic:(NSString *)pushTopic
{
    bool tooMany = false;

    @synchronized(self) {
        if ([self.taskMap count] > 5) {
            self.lostEvents++;
            tooMany = true;
        }
    }
    if (tooMany) {
        os_log(OS_LOG_DEFAULT, "metrics %@ dropped on floor since too many are pending", event.eventName);
        return;
    }

    SECC2MPGenericEvent *genericEvent = [event genericEvent];
    if (genericEvent == NULL) {
        return;
    }

    NSMutableURLRequest *request = [self requestForGenericEvent:genericEvent];
    if (request == NULL) {
        return;
    }

    NSURLSessionDataTask *task = [self.URLSession dataTaskWithRequest:request];
    if (pushTopic) {
#if !TARGET_OS_BRIDGE
        task._APSRelayTopic = pushTopic;
#endif
    }

    @synchronized(self) {
        if ([self.taskMap count] == 0) {
            self.transaction = os_transaction_create("com.apple.security.c2metric.upload");
        }
        self.taskMap[@(task.taskIdentifier)] = event;
    }

    [task resume];
}

- (SecEventMetric *)getEvent:(NSURLSessionTask *)task
{
    @synchronized(self) {
        return self.taskMap[@(task.taskIdentifier)];
    }
}

//MARK: - URLSession Callbacks

- (void)    URLSession:(NSURLSession *)session
                  task:(NSURLSessionTask *)task
  didCompleteWithError:(nullable NSError *)error
{
    SecEventMetric *event = [self getEvent:task];

    os_log(OS_LOG_DEFAULT, "metrics %@ transfer %@ completed with: %@",
           event.eventName, task.originalRequest.URL, error ? [error description] : @"success");

    @synchronized(self) {
        [self.taskMap removeObjectForKey:@(task.taskIdentifier)];
        if (self.lostEvents || error) {
            NSMutableDictionary *event = [NSMutableDictionary dictionary];

            if (self.lostEvents) {
                event[@"counter"] = @(self.lostEvents);
            }
            if (error) {
                event[@"error_code"] = @(error.code);
                event[@"error_domain"] = error.domain;
            }
            [SecCoreAnalytics sendEvent:@"com.apple.security.push.channel.dropped" event:event];
            self.lostEvents = 0;
        }

        if (self.taskMap.count == 0) {
            self.transaction = NULL;
        }
    }
}

//MARK: - FOO


- (NSMutableURLRequest*)requestForGenericEvent:(SECC2MPGenericEvent*)genericEvent
{
    NSURL* metricURL = [[self class] c2MetricsEndpoint];
    if (!metricURL) {
        return nil;
    }

    NSMutableURLRequest* request = [[NSMutableURLRequest alloc] initWithURL:metricURL];
    if (!request) {
        return nil;
    }

    SECC2MPMetric* metrics = [[SECC2MPMetric alloc] init];
    if (!metrics) {
        return nil;
    }
    metrics.deviceInfo = [self generateDeviceInfo];
    metrics.reportFrequency = 0;
    metrics.reportFrequencyBase = 0;

    metrics.metricType = SECC2MPMetric_Type_generic_event_type;
    metrics.genericEvent = genericEvent;

    PBDataWriter* protobufWriter = [[PBDataWriter alloc] init];
    if (!protobufWriter) {
        return nil;
    }
    [metrics writeTo:protobufWriter];
    NSData* metricData = [protobufWriter immutableData];
    if (!metricData) {
        return nil;
    }
    NSData* compressedData = [self gzipEncode:metricData];
    if (!compressedData) {
        return nil;
    }
    [request setHTTPMethod:@"POST"];
    [request setHTTPBody:compressedData];
    [request setValue:@"application/protobuf" forHTTPHeaderField:@"Content-Type"];
    [request setValue:@"gzip" forHTTPHeaderField:@"Content-Encoding"];

    return request;
}

#define CHUNK 1024

- (NSData*) gzipEncode:(NSData*)bodyData {
    unsigned have;
    unsigned char outBytes[CHUNK] = {0};
    NSMutableData *compressedData = [NSMutableData data];

    /* allocate deflate state */
    z_stream _zlibStream;
    _zlibStream.zalloc = Z_NULL;
    _zlibStream.zfree = Z_NULL;
    _zlibStream.opaque = Z_NULL;

    // generate gzip header/trailer, use defaults for all other values
    int ret = deflateInit2(&_zlibStream, Z_DEFAULT_COMPRESSION, Z_DEFLATED, 15 + 16, 8, Z_DEFAULT_STRATEGY);
    if (ret != Z_OK) {
        return NULL;
    }

    NS_VALID_UNTIL_END_OF_SCOPE NSData *arcSafeBodyData = bodyData;
    _zlibStream.next_in = (Bytef *)[arcSafeBodyData bytes];
    _zlibStream.avail_in = (unsigned int)[arcSafeBodyData length];
    do {
        _zlibStream.avail_out = CHUNK;
        _zlibStream.next_out = outBytes;
        ret = deflate(&_zlibStream, Z_FINISH);
        assert(ret != Z_STREAM_ERROR);
        have = CHUNK - _zlibStream.avail_out;
        [compressedData appendBytes:outBytes length:have];
    } while (_zlibStream.avail_out == 0);
    assert(_zlibStream.avail_in == 0);
    deflateEnd(&_zlibStream);

    return compressedData;
}


- (SECC2MPDeviceInfo*) generateDeviceInfo {
    SECC2MPDeviceInfo* deviceInfo = [[SECC2MPDeviceInfo alloc] init];
    deviceInfo.productName = [SecC2DeviceInfo productName];
    deviceInfo.productType = [SecC2DeviceInfo productType];
    deviceInfo.productVersion = [SecC2DeviceInfo productVersion];
    deviceInfo.productBuild = [SecC2DeviceInfo buildVersion];
    deviceInfo.processName = [SecC2DeviceInfo processName];
    deviceInfo.processVersion = [SecC2DeviceInfo processVersion];
    deviceInfo.processUuid = [SecC2DeviceInfo processUUID];
    return deviceInfo;
}

@end
