/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 8, 2023.
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

@protocol RPProto
- (nullable id)accessorInProto;
@end

@interface RPFoo <RPProto>
@property (readonly, nonnull) int *nonnullToNullable;
@property (readonly, nullable) int *nullableToNonnull;
@property (readonly, nonnull) id typeChangeMoreSpecific;
@property (readonly, nonnull) RPFoo *typeChangeMoreGeneral;

@property (readonly, nonnull) id accessorRedeclaredAsNullable;
- (nullable id)accessorRedeclaredAsNullable;

- (nullable id)accessorDeclaredFirstAsNullable;
@property (readonly, nonnull) id accessorDeclaredFirstAsNullable;

@property (readonly, nullable) id accessorInProto;
@end

@interface RPBase <RPProto>
@property (readonly, nonatomic, nullable) id accessorInProto;
@end
