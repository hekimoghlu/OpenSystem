/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 12, 2024.
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

NS_ASSUME_NONNULL_BEGIN

/*
 * This class holds a set of weak pointers to 'listener' objects, and offers the chance to dispatch updates
 * to them on each listener's own serial dispatch queue
 */

@interface CKKSListenerCollection<__covariant ListenerType> : NSObject
- (instancetype)init NS_UNAVAILABLE;
- (instancetype)initWithName:(NSString*)name;

- (void)registerListener:(ListenerType)listener;
- (void)iterateListeners:(void (^)(ListenerType))block;
@end

NS_ASSUME_NONNULL_END
