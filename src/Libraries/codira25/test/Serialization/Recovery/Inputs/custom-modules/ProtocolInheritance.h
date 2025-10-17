/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 2, 2024.
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

@protocol Order2_ConsistentBaseProto
- (void)consistent;
@end

@protocol Order4_ConsistentBaseProto
- (void)consistent;
@end

@protocol Order1_FickleBaseProto
- (void)fickle;
@optional
- (void)extraFickle;
@end

@protocol Order3_FickleBaseProto
- (void)fickle;
@optional
- (void)extraFickle;
@end

@protocol Order5_FickleBaseProto
- (void)fickle;
@optional
- (void)extraFickle;
@end

// The actual order here is determined by the protocol names.
#if EXTRA_PROTOCOL_FIRST
@protocol SubProto <Order1_FickleBaseProto, Order2_ConsistentBaseProto, Order4_ConsistentBaseProto>
@end
#elif EXTRA_PROTOCOL_MIDDLE
@protocol SubProto <Order2_ConsistentBaseProto, Order3_FickleBaseProto, Order4_ConsistentBaseProto>
@end
#elif EXTRA_PROTOCOL_LAST
@protocol SubProto <Order2_ConsistentBaseProto, Order4_ConsistentBaseProto, Order5_FickleBaseProto>
@end
#elif NO_EXTRA_PROTOCOLS
@protocol SubProto <Order2_ConsistentBaseProto, Order4_ConsistentBaseProto>
@end
#else
# error "Missing -D flag"
#endif
