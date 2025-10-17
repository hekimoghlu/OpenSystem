/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 29, 2024.
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


typedef __INTPTR_TYPE__ intptr_t;

__attribute__((objc_root_class))
@interface Base
- (instancetype)init;
@end

@interface IncompleteDesignatedInitializers : Base
- (instancetype)initFirst:(intptr_t)x __attribute__((objc_designated_initializer));
- (instancetype)initSecond:(intptr_t)x __attribute__((objc_designated_initializer));
- (instancetype)initMissing:(intptr_t)x, ... __attribute__((objc_designated_initializer));
- (instancetype)initConveniently:(intptr_t)x;
@end
@interface IncompleteDesignatedInitializers (CategoryConvenience)
- (instancetype)initCategory:(intptr_t)x;
@end

@interface IncompleteConvenienceInitializers : Base
- (instancetype)initFirst:(intptr_t)x __attribute__((objc_designated_initializer));
- (instancetype)initSecond:(intptr_t)x __attribute__((objc_designated_initializer));
- (instancetype)initMissing:(intptr_t)x, ...;
- (instancetype)initConveniently:(intptr_t)x;
@end
@interface IncompleteConvenienceInitializers (CategoryConvenience)
- (instancetype)initCategory:(intptr_t)x;
@end

@interface IncompleteUnknownInitializers : Base
- (instancetype)initFirst:(intptr_t)x;
- (instancetype)initSecond:(intptr_t)x;
- (instancetype)initMissing:(intptr_t)x, ...;
- (instancetype)initConveniently:(intptr_t)x;
@end
@interface IncompleteUnknownInitializers (CategoryConvenience)
- (instancetype)initCategory:(intptr_t)x;
@end

@interface IncompleteDesignatedInitializersWithCategory : Base
- (instancetype)initFirst:(intptr_t)x __attribute__((objc_designated_initializer));
- (instancetype)initMissing:(intptr_t)x, ... __attribute__((objc_designated_initializer));
- (instancetype)initConveniently:(intptr_t)x;
@end
@interface IncompleteDesignatedInitializersWithCategory (/*class extension*/)
- (instancetype)initSecond:(intptr_t)x __attribute__((objc_designated_initializer));
- (instancetype)initCategory:(intptr_t)x;
@end

@interface DesignatedInitializerInAnotherModule : Base
- (instancetype)initFirst:(intptr_t)x __attribute__((objc_designated_initializer));
- (instancetype)initSecond:(intptr_t)x __attribute__((objc_designated_initializer));
- (instancetype)initMissing:(intptr_t)x, ... __attribute__((objc_designated_initializer));
- (instancetype)initConveniently:(intptr_t)x;
@end
@interface DesignatedInitializerInAnotherModule (CategoryConvenience)
- (instancetype)initCategory:(intptr_t)x;
@end
