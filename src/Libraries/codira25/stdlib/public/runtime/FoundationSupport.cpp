/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 30, 2024.
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

//===--- FoundationSupport.cpp - Support functions for Foundation ---------===//
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
// Helper functions for the Foundation framework.
//
//===----------------------------------------------------------------------===//

#include "language/Runtime/FoundationSupport.h"

#if LANGUAGE_OBJC_INTEROP
#include "language/Runtime/Metadata.h"
#include "language/Runtime/HeapObject.h"

using namespace language;

/// Returns a boolean indicating whether the Objective-C name of a class type is
/// stable across executions, i.e., if the class name is safe to serialize. (The
/// names of private and local types are unstable.)
bool
language::_language_isObjCTypeNameSerializable(Class theClass) {
  auto type = (AnyClassMetadata *)theClass;
  switch (type->getKind()) {
  case MetadataKind::ObjCClassWrapper:
  case MetadataKind::ForeignClass:
  case MetadataKind::ForeignReferenceType:
    return true;
  case MetadataKind::Class: {
    // Pure ObjC classes always have stable names.
    if (type->isPureObjC())
      return true;
    auto cls = static_cast<const ClassMetadata *>(type);
    // Peek through artificial subclasses.
    if (cls->isArtificialSubclass()) {
      cls = cls->Superclass;
    }
    // A custom ObjC name is always considered stable.
    if (cls->getFlags() & ClassFlags::HasCustomObjCName)
      return true;
    // Otherwise the name is stable if the class has no anonymous ancestor context.
    auto desc = static_cast<const ContextDescriptor *>(cls->getDescription());
    while (desc) {
      if (desc->getKind() == ContextDescriptorKind::Anonymous) {
        return false;
      }
      desc = desc->Parent.get();
    }
    return true;
  }
  default:
    return false;
  }
}
#endif // LANGUAGE_OBJC_INTEROP
