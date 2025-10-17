/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 26, 2024.
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

@interface Object
- (nonnull instancetype)init;
@end

@interface Base : Object
#if !BAD
- (void)disappearingMethod;
- (nullable id)nullabilityChangeMethod;
- (nonnull id)typeChangeMethod;
@property (readonly) long disappearingProperty;
@property (readwrite) long disappearingPropertySetter;
#else
//- (void)disappearingMethod;
- (nonnull id)nullabilityChangeMethod;
- (nonnull Base *)typeChangeMethod;
// @property (readonly) long disappearingProperty;
@property (readonly) long disappearingPropertySetter;
#endif
@end

@interface Base (ExtraMethodsToTriggerCircularReferences)
#if !BAD
- (void)disappearingMethodWithOverload;
#else
//- (void)disappearingMethodWithOverload;
#endif
@end

@interface GenericBase<Element>: Object
#if !BAD
- (void)disappearingMethod;
- (nullable Element)nullabilityChangeMethod;
- (nonnull id)typeChangeMethod;
#else
//- (void)disappearingMethod;
- (nonnull Element)nullabilityChangeMethod;
- (nonnull Element)typeChangeMethod;
#endif
@end


@interface IndexedSubscriptDisappearsBase : Object
#if !BAD
- (nonnull id)objectAtIndexedSubscript:(long)index;
#else
//- (nonnull id)objectAtIndexedSubscript:(long)index;
#endif
@end

@interface KeyedSubscriptDisappearsBase : Object
#if !BAD
- (nonnull id)objectForKeyedSubscript:(nonnull id)key;
#else
// - (nonnull id)objectForKeyedSubscript:(nonnull id)key;
#endif
@end

@interface GenericIndexedSubscriptDisappearsBase<Element> : Object
#if !BAD
- (nonnull Element)objectAtIndexedSubscript:(long)index;
#else
// - (nonnull Element)objectAtIndexedSubscript:(long)index;
#endif
@end

@interface GenericKeyedSubscriptDisappearsBase<Element> : Object
#if !BAD
- (nonnull Element)objectAtIndexedSubscript:(nonnull id)key;
#else
// - (nonnull Element)objectAtIndexedSubscript:(nonnull id)key;
#endif
@end


@interface DesignatedInitDisappearsBase : Object
- (nonnull instancetype)init __attribute__((objc_designated_initializer));
- (nonnull instancetype)initConvenience:(long)value;
#if !BAD
- (nonnull instancetype)initWithValue:(long)value __attribute__((objc_designated_initializer));
#else
//- (nonnull instancetype)initWithValue:(long)value __attribute__((objc_designated_initializer));
#endif
@end

@interface OnlyDesignatedInitDisappearsBase : Object
- (nonnull instancetype)initConvenience:(long)value;
#if !BAD
- (nonnull instancetype)initWithValue:(long)value __attribute__((objc_designated_initializer));
#else
//- (nonnull instancetype)initWithValue:(long)value __attribute__((objc_designated_initializer));
#endif
@end

@interface ConvenienceInitDisappearsBase : Object
- (nonnull instancetype)init __attribute__((objc_designated_initializer));
- (nonnull instancetype)initConvenience:(long)value;
#if !BAD
- (nonnull instancetype)initWithValue:(long)value;
#else
//- (nonnull instancetype)initWithValue:(long)value;
#endif
@end

@interface UnknownInitDisappearsBase : Object
- (nonnull instancetype)init;
#if !BAD
- (nonnull instancetype)initWithValue:(long)value;
#else
//- (nonnull instancetype)initWithValue:(long)value;
#endif
@end

@interface OnlyUnknownInitDisappearsBase : Object
#if !BAD
- (nonnull instancetype)initWithValue:(long)value;
#else
//- (nonnull instancetype)initWithValue:(long)value;
#endif
@end


#if !BAD
struct BoxedInt {
  int value;
};
#endif

@interface MethodWithDisappearingType : Object
#if !BAD
- (struct BoxedInt)boxItUp;
#endif
@end

@interface InitializerWithDisappearingType : Object
#if !BAD
- (nonnull instancetype)initWithBoxedInt:(struct BoxedInt)box;
#endif
@end
