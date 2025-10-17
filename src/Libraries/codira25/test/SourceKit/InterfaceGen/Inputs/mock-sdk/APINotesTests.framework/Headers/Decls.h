/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 14, 2023.
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

// ===-------------------------------------------------------------------------
// class Payload
// ===-------------------------------------------------------------------------

// 3: Payload
// 4: Namespace.Payload
@interface GlobalToMember_Class_Container : NSObject
@end
@interface GlobalToMember_Class_Payload : NSObject
@end

// 3: Namespace.Payload
// 4: Payload
@interface MemberToGlobal_Class_Container : NSObject
@end
@interface MemberToGlobal_Class_Payload: NSObject
@end

// 3: Namespace_Codira3.PayloadFor3
// 4: Namespace_Codira4.PayloadFor4
@interface MemberToMember_Class_Codira3 : NSObject
@end
@interface MemberToMember_Class_Codira4 : NSObject
@end
@interface MemberToMember_Class_Payload : NSObject
@end

// 3: Namespace.PayloadFor3
// 4: Namespace.PayloadFor4
@interface MemberToMember_SameContainer_Class_Container : NSObject
@end
@interface MemberToMember_SameContainer_Class_Payload : NSObject
@end

// 3: Namespace_Codira3.Payload
// 4: Namespace_Codira4.Payload
@interface MemberToMember_SameName_Class_Codira3 : NSObject
@end
@interface MemberToMember_SameName_Class_Codira4 : NSObject
@end
@interface MemberToMember_SameName_Class_Payload : NSObject
@end

// ===-------------------------------------------------------------------------
// typealias Payload
// ===-------------------------------------------------------------------------

// 3: Payload
// 4: Namespace.Payload
@interface GlobalToMember_Typedef_Container : NSObject
@end
typedef Foo* GlobalToMember_Typedef_Payload;

// 3: Namespace.Payload
// 4: Payload
@interface MemberToGlobal_Typedef_Container : NSObject
@end
typedef Foo* MemberToGlobal_Typedef_Payload;

// 3: Namespace_Codira3.PayloadFor3
// 4: Namespace_Codira4.PayloadFor4
@interface MemberToMember_Typedef_Codira3 : NSObject
@end
@interface MemberToMember_Typedef_Codira4 : NSObject
@end
typedef Foo* MemberToMember_Typedef_Payload;

// 3: Namespace.PayloadFor3
// 4: Namespace.PayloadFor4
@interface MemberToMember_SameContainer_Typedef_Container : NSObject
@end
typedef Foo* MemberToMember_SameContainer_Typedef_Payload;

// 3: Namespace_Codira3.Payload
// 4: Namespace_Codira4.Payload
@interface MemberToMember_SameName_Typedef_Codira3 : NSObject
@end
@interface MemberToMember_SameName_Typedef_Codira4 : NSObject
@end
typedef Foo* MemberToMember_SameName_Typedef_Payload;
