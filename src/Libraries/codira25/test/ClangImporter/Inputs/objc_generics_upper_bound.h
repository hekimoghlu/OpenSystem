/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 25, 2025.
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

@protocol P
@end

@interface C: NSObject
@end

@interface Base<T> : NSObject
@end

@interface Derived: Base
@end

@interface BaseWithProtocol<T: id<P>> : NSObject
@end

@interface DerivedWithProtocol: BaseWithProtocol
@end

@interface BaseWithClass<T: C *> : NSObject
@end

@interface DerivedWithClass: BaseWithClass
@end

@interface BaseWithBoth<T: C<P> *> : NSObject
@end

@interface DerivedWithBoth: BaseWithBoth
@end
