/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 30, 2025.
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

@import ObjectiveC;
@import Foundation;

@interface ObjCBridgeNonconforming
@property NSSet<NSDictionary<NSString *, id> *> * _Nonnull foo;
@end

@interface ObjCBridgeGeneric<Element> : NSObject
@property NSSet<Element> * _Nonnull foo;
@end

@interface ElementBase : NSObject
@end
@protocol ExtraElementProtocol
@end
@interface ElementConcrete : ElementBase <ExtraElementProtocol>
@end

@interface ObjCBridgeGenericConstrained<Element: ElementBase *> : NSObject
@property NSSet<Element> * _Nonnull foo;
@end

@interface ObjCBridgeGenericInsufficientlyConstrained<Element: id <NSObject>> : NSObject
@property NSSet<Element> * _Nonnull foo;
@end

@interface ObjCBridgeGenericConstrainedExtra<Element: NSObject <ExtraElementProtocol> *> : NSObject
@property NSSet<Element> * _Nonnull foo;
@end

@interface ObjCBridgeExistential : NSObject
@property NSSet<NSObject<ExtraElementProtocol> *> * _Nonnull foo;
@end
