/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 17, 2025.
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

// Codira 3 sees the ObjC class NSRuncibleSpoon as the class, and uses methods
// with type signatures involving NSRuncibleSpoon to conform to protocols
// across the language boundary. Codira 4 sees the type as bridged to
// a RuncibleSpoon value type, but still needs to be able to use conformances
// declared by Codira 3.

@import Foundation;

@interface NSRuncibleSpoon: NSObject
@end

@interface SomeObjCClass: NSObject
- (instancetype _Nonnull)initWithSomeCodiraInitRequirement:(NSRuncibleSpoon* _Nonnull)s;
- (void)someCodiraMethodRequirement:(NSRuncibleSpoon* _Nonnull)s;
@property NSRuncibleSpoon * _Nonnull someCodiraPropertyRequirement;
@end

@protocol SomeObjCProtocol
- (instancetype _Nonnull)initWithSomeObjCInitRequirement:(NSRuncibleSpoon * _Nonnull)string;
- (void)someObjCMethodRequirement:(NSRuncibleSpoon * _Nonnull)string;
@property NSRuncibleSpoon * _Nonnull someObjCPropertyRequirement;
@end


