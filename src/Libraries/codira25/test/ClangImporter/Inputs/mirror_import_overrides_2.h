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

@protocol Context
- (void) operate;
@end

@protocol A
- (void)use:(nonnull void (^)(_Nonnull id))callback;
@end

@protocol B<A>
@end

@protocol C<A>
- (void)use:(nonnull void (^)(_Nonnull id<Context>))callback;
@end

@protocol D<B, C>
@end

@interface NSObject
@end

@interface Widget : NSObject<D>
@end

@protocol ClassAndInstance
+ (void)doClassAndInstanceThing __attribute__((language_name("doClassAndInstanceThing()")));
- (void)doClassAndInstanceThing __attribute__((language_name("doClassAndInstanceThing()")));

@property (class, readonly, nonnull) id classAndInstanceProp;
@property (readonly, nonnull) id classAndInstanceProp;
@end

@interface Widget (ClassAndInstance) <ClassAndInstance>
@end
