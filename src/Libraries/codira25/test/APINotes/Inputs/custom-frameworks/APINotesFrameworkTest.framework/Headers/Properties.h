/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 22, 2022.
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

#ifdef __OBJC__
#pragma clang assume_nonnull begin

@interface TestProperties: Base
@property (nonatomic, readwrite, retain) id accessorsOnly;
@property (nonatomic, readwrite, retain, class) id accessorsOnlyForClass;

@property (nonatomic, readonly, retain) id accessorsOnlyRO;
@property (nonatomic, readwrite, weak) id accessorsOnlyWeak;

@property (nonatomic, readwrite, retain) id accessorsOnlyInVersion4;
@property (nonatomic, readwrite, retain, class) id accessorsOnlyForClassInVersion4;

@property (nonatomic, readwrite, retain) id accessorsOnlyExceptInVersion4;
@property (nonatomic, readwrite, retain, class) id accessorsOnlyForClassExceptInVersion4;
@end

@interface TestPropertiesSub: TestProperties
@property (nonatomic, readwrite, retain) id accessorsOnly;
@property (nonatomic, readwrite, retain, class) id accessorsOnlyForClass;
@end

@interface TestProperties (Retyped)
@property (nonatomic, readwrite, retain) id accessorsOnlyWithNewType;
@end

@interface TestProperties (AccessorsOnlyCustomized)
@property (nonatomic, readwrite, retain, null_resettable) id accessorsOnlyRenamedRetyped;
@property (class, nonatomic, readwrite, retain, null_resettable) id accessorsOnlyRenamedRetypedClass;
@end

#pragma clang assume_nonnull end
#endif // __OBJC__
