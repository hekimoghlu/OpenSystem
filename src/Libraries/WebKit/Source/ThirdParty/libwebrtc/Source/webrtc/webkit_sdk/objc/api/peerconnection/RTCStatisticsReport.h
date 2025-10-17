/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 23, 2022.
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
#import <Foundation/Foundation.h>

@class RTCStatistics;

NS_ASSUME_NONNULL_BEGIN

/** A statistics report. Encapsulates a number of RTCStatistics objects. */
@interface RTCStatisticsReport : NSObject

/** The timestamp of the report in microseconds since 1970-01-01T00:00:00Z. */
@property(nonatomic, readonly) CFTimeInterval timestamp_us;

/** RTCStatistics objects by id. */
@property(nonatomic, readonly) NSDictionary<NSString *, RTCStatistics *> *statistics;

- (instancetype)init NS_UNAVAILABLE;

@end

/** A part of a report (a subreport) covering a certain area. */
@interface RTCStatistics : NSObject

/** The id of this subreport, e.g. "RTCMediaStreamTrack_receiver_2". */
@property(nonatomic, readonly) NSString *id;

/** The timestamp of the subreport in microseconds since 1970-01-01T00:00:00Z. */
@property(nonatomic, readonly) CFTimeInterval timestamp_us;

/** The type of the subreport, e.g. "track", "codec". */
@property(nonatomic, readonly) NSString *type;

/** The keys and values of the subreport, e.g. "totalFramesDuration = 5.551".
    The values are either NSNumbers or NSStrings, or NSArrays encapsulating NSNumbers
    or NSStrings. */
@property(nonatomic, readonly) NSDictionary<NSString *, NSObject *> *values;

- (instancetype)init NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END
