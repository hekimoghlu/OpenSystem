/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 2, 2023.
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
#if OCTAGON

#import <Foundation/Foundation.h>
#import "CKKSOutgoingQueueEntry.h"

NS_ASSUME_NONNULL_BEGIN

@interface CKKSRateLimiter : NSObject <NSSecureCoding>

@property (readonly) NSDictionary* config;  // of NSString : NSNumber

/*!
 * @brief Find out whether outgoing items are okay to send.
 * @param entry The outgoing object being judged.
 * @param time Current time.
 * @param limitTime In case of badness, this will contain the time at which the object may be sent.
 * @return Badness score from 0 (fine to send immediately) to 5 (overload, keep back), or -1 in case caller does not provide an NSDate object.
 *
 * judge:at: will set the limitTime object to nil in case of 0 badness. For badnesses 1-4 the time object will indicate when it is okay to send the entry.
 * At badness 5 judge:at: has determined there is too much activity so the caller should hold off altogether. The limitTime object will indicate when
 * this overloaded state will end.
 */
- (int)judge:(CKKSOutgoingQueueEntry* const)entry
           at:(NSDate*)time
    limitTime:(NSDate* _Nonnull __autoreleasing* _Nonnull)limitTime;

- (instancetype)init;
- (instancetype _Nullable)initWithCoder:(NSCoder* _Nullable)coder NS_DESIGNATED_INITIALIZER;
- (NSUInteger)stateSize;
- (void)reset;
- (NSString*)diagnostics;

+ (BOOL)supportsSecureCoding;

@end

NS_ASSUME_NONNULL_END
#endif
