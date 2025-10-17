/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 8, 2023.
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

#import "RTCMacros.h"

NS_ASSUME_NONNULL_BEGIN

RTC_OBJC_EXPORT
@protocol RTCDtmfSender <NSObject>

/**
 * Returns true if this RTCDtmfSender is capable of sending DTMF. Otherwise
 * returns false. To be able to send DTMF, the associated RTCRtpSender must be
 * able to send packets, and a "telephone-event" codec must be negotiated.
 */
@property(nonatomic, readonly) BOOL canInsertDtmf;

/**
 * Queues a task that sends the DTMF tones. The tones parameter is treated
 * as a series of characters. The characters 0 through 9, A through D, #, and *
 * generate the associated DTMF tones. The characters a to d are equivalent
 * to A to D. The character ',' indicates a delay of 2 seconds before
 * processing the next character in the tones parameter.
 *
 * Unrecognized characters are ignored.
 *
 * @param duration The parameter indicates the duration to use for each
 * character passed in the tones parameter. The duration cannot be more
 * than 6000 or less than 70 ms.
 *
 * @param interToneGap The parameter indicates the gap between tones.
 * This parameter must be at least 50 ms but should be as short as
 * possible.
 *
 * If InsertDtmf is called on the same object while an existing task for this
 * object to generate DTMF is still running, the previous task is canceled.
 * Returns true on success and false on failure.
 */
- (BOOL)insertDtmf:(nonnull NSString *)tones
          duration:(NSTimeInterval)duration
      interToneGap:(NSTimeInterval)interToneGap;

/** The tones remaining to be played out */
- (nonnull NSString *)remainingTones;

/**
 * The current tone duration value. This value will be the value last set via the
 * insertDtmf method, or the default value of 100 ms if insertDtmf was never called.
 */
- (NSTimeInterval)duration;

/**
 * The current value of the between-tone gap. This value will be the value last set
 * via the insertDtmf() method, or the default value of 50 ms if insertDtmf() was never
 * called.
 */
- (NSTimeInterval)interToneGap;

@end

NS_ASSUME_NONNULL_END
