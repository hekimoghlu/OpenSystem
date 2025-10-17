/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 20, 2025.
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
#import "RTCLegacyStatsReport+Private.h"

#import "base/RTCLogging.h"
#import "helpers/NSString+StdString.h"

#include "rtc_base/checks.h"

@implementation RTCLegacyStatsReport

@synthesize timestamp = _timestamp;
@synthesize type = _type;
@synthesize reportId = _reportId;
@synthesize values = _values;

- (NSString *)description {
  return [NSString stringWithFormat:@"RTCLegacyStatsReport:\n%@\n%@\n%f\n%@",
                                    _reportId,
                                    _type,
                                    _timestamp,
                                    _values];
}

#pragma mark - Private

- (instancetype)initWithNativeReport:(const webrtc::StatsReport &)nativeReport {
  if (self = [super init]) {
    _timestamp = nativeReport.timestamp();
    _type = [NSString stringForStdString:nativeReport.TypeToString()];
    _reportId = [NSString stringForStdString:
        nativeReport.id()->ToString()];

    NSUInteger capacity = nativeReport.values().size();
    NSMutableDictionary *values =
        [NSMutableDictionary dictionaryWithCapacity:capacity];
    for (auto const &valuePair : nativeReport.values()) {
      NSString *key = [NSString stringForStdString:
          valuePair.second->display_name()];
      NSString *value = [NSString stringForStdString:
          valuePair.second->ToString()];

      // Not expecting duplicate keys.
      RTC_DCHECK(![values objectForKey:key]);
      [values setObject:value forKey:key];
    }
    _values = values;
  }
  return self;
}

@end
