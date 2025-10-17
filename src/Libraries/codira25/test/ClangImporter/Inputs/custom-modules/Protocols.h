/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 24, 2022.
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

@protocol FooProto
@property int bar;
@end

@protocol AnotherProto
@end

Class <FooProto> _Nonnull processFooType(Class <FooProto> _Nonnull);
Class <FooProto, AnotherProto> _Nonnull processComboType(Class <FooProto, AnotherProto> _Nonnull);
Class <AnotherProto, FooProto> _Nonnull processComboType2(Class <AnotherProto, FooProto> _Nonnull);


@protocol SubProto <FooProto>
@end

@interface ProtocolTestingBase
@end

@interface SubProtoImpl: ProtocolTestingBase <SubProto>
@end
