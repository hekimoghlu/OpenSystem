/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 18, 2024.
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
#ifndef SFAnalyticsDefines_h
#define SFAnalyticsDefines_h

#if __OBJC2__

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

extern NSString* const SFAnalyticsTableSuccessCount;
extern NSString* const SFAnalyticsTableHardFailures;
extern NSString* const SFAnalyticsTableSoftFailures;
extern NSString* const SFAnalyticsTableSamples;
extern NSString* const SFAnalyticsTableNotes;
extern NSString* const SFAnalyticsTableRockwell;

extern NSString* const SFAnalyticsColumnSuccessCount;
extern NSString* const SFAnalyticsColumnHardFailureCount;
extern NSString* const SFAnalyticsColumnSoftFailureCount;
extern NSString* const SFAnalyticsColumnSampleValue;
extern NSString* const SFAnalyticsColumnSampleName;

extern NSString* const SFAnalyticsPostTime;
extern NSString* const SFAnalyticsEventTime;
extern NSString* const SFAnalyticsEventType;
extern NSString* const SFAnalyticsEventTypeErrorEvent;
extern NSString* const SFAnalyticsEventErrorDestription;
extern NSString* const SFAnalyticsEventClassKey;

// Helpers for logging NSErrors
extern NSString* const SFAnalyticsAttributeErrorUnderlyingChain;
extern NSString* const SFAnalyticsAttributeErrorDomain;
extern NSString* const SFAnalyticsAttributeErrorCode;

extern NSString* const SFAnalyticsAttributeLastUploadTime;

extern NSString* const SFAnalyticsUserDefaultsSuite;

extern char* const SFAnalyticsFireSamplersNotification;

/* Internal Topic Names */
extern NSString* const SFAnalyticsTopicCloudServices;
extern NSString* const SFAnalyticsTopicKeySync;
extern NSString* const SFAnalyticsTopicTrust;
extern NSString* const SFAnalyticsTopicTransparency;
extern NSString* const SFAnalyticsTopicSWTransparency;
extern NSString* const SFAnalyticsTopicNetworking;

typedef NS_ENUM(NSInteger, SFAnalyticsEventClass) {
    SFAnalyticsEventClassSuccess = 0,
    SFAnalyticsEventClassHardFailure,
    SFAnalyticsEventClassSoftFailure,
    SFAnalyticsEventClassNote,
    SFAnalyticsEventClassRockwell,
};

extern NSString* const SFAnalyticsTableSchema;

// We can only send this many events in total to splunk per upload
extern NSUInteger const SFAnalyticsMaxEventsToReport;

extern NSString* const SFAnalyticsErrorDomain;

NS_ASSUME_NONNULL_END

#endif /* __OBJC2__ */

#endif /* SFAnalyticsDefines_h */
