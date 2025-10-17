/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 18, 2022.
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

@import Foundation;

#define LANGUAGE_NAME(X) __attribute__((language_name(#X)))

#pragma clang assume_nonnull begin

@interface Test : NSObject

// "Factory methods" that we'd rather have as initializers.
+ (instancetype)a LANGUAGE_NAME(init());
+ (instancetype)b LANGUAGE_NAME(init(dummyParam:));

+ (instancetype)c:(id)x LANGUAGE_NAME(init(cc:));
+ (instancetype)d:(id)x LANGUAGE_NAME(init(_:));

+ (instancetype)e:(id)a e:(id)b e:(id)c LANGUAGE_NAME(init(aa:_:cc:));

+ (Test *)f LANGUAGE_NAME(init(fixedType:));

// Would-be initializers.
+ (instancetype)testZ LANGUAGE_NAME(zz());
+ (instancetype)testY:(id)x LANGUAGE_NAME(yy(aa:));
+ (instancetype)testX:(id)x xx:(id)xx LANGUAGE_NAME(xx(_:bb:));

@end

@interface TestError : NSObject
// Factory methods with NSError.
+ (nullable instancetype)err1:(NSError **)err LANGUAGE_NAME(init(error:));
+ (nullable instancetype)err2:(nullable id)x error:(NSError **)err LANGUAGE_NAME(init(aa:error:));
+ (nullable instancetype)err3:(nullable id)x error:(NSError **)err callback:(void(^)(void))block LANGUAGE_NAME(init(aa:error:block:));
+ (nullable instancetype)err4:(NSError **)err callback:(void(^)(void))block LANGUAGE_NAME(init(error:block:));

+ (nullable instancetype)err5:(nullable id)x error:(NSError **)err LANGUAGE_NAME(init(aa:));
+ (nullable instancetype)err6:(nullable id)x error:(NSError **)err callback:(void(^)(void))block LANGUAGE_NAME(init(aa:block:));
+ (nullable instancetype)err7:(NSError **)err callback:(void(^)(void))block LANGUAGE_NAME(init(block:));

// Would-be initializers.
+ (nullable instancetype)testW:(nullable id)x error:(NSError **)err LANGUAGE_NAME(ww(_:));
+ (nullable instancetype)testW2:(nullable id)x error:(NSError **)err LANGUAGE_NAME(w2(_:error:));
+ (nullable instancetype)testV:(NSError **)err LANGUAGE_NAME(vv());
+ (nullable instancetype)testV2:(NSError **)err LANGUAGE_NAME(v2(error:));
@end

@interface TestSub : Test
@end

@interface TestErrorSub : TestError
@end

#pragma clang assume_nonnull end
