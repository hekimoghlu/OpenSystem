/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 21, 2022.
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

//===--- ImplementTypeIDZone.h - Implement a TypeID Zone --------*- C++ -*-===//
//
// Copyright (c) NeXTHub Corporation. All rights reserved.
// DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
//
// This code is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
// version 2 for more details (a copy is included in the LICENSE file that
// accompanied this code).
//
// Author(-s): Tunjay Akbarli
//

//===----------------------------------------------------------------------===//
//
//  This file should be #included to implement the TypeIDs for a given zone
//  in a C++ file.
//  Two macros should be #define'd before inclusion, and will be #undef'd at
//  the end of this file:
//
//    LANGUAGE_TYPEID_ZONE: The ID number of the Zone being defined, which must
//    be unique. 0 is reserved for basic C and LLVM types; 255 is reserved
//    for test cases.
//
//    LANGUAGE_TYPEID_HEADER: A (quoted) name of the header to be
//    included to define the types in the zone.
//
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_TYPEID_ZONE
#  error Must define the value of the TypeID zone with the given name.
#endif

#ifndef LANGUAGE_TYPEID_HEADER
#  error Must define the TypeID header name with LANGUAGE_TYPEID_HEADER
#endif

// Define a TypeID where the type name and internal name are the same.
#define LANGUAGE_REQUEST(Zone, Type, Sig, Caching, LocOptions) LANGUAGE_TYPEID_NAMED(Type, Type)
#define LANGUAGE_TYPEID(Type) LANGUAGE_TYPEID_NAMED(Type, Type)

// Out-of-line definitions.
#define LANGUAGE_TYPEID_NAMED(Type, Name)            \
  const uint64_t TypeID<Type>::value;

#define LANGUAGE_TYPEID_TEMPLATE1_NAMED(Template, Name, Param1, Arg1)

#include LANGUAGE_TYPEID_HEADER

#undef LANGUAGE_REQUEST

#undef LANGUAGE_TYPEID_NAMED
#undef LANGUAGE_TYPEID_TEMPLATE1_NAMED

#undef LANGUAGE_TYPEID
#undef LANGUAGE_TYPEID_ZONE
#undef LANGUAGE_TYPEID_HEADER
