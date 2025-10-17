/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 3, 2022.
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

#ifndef TEST_INTEROP_CXX_NAMESPACE_INPUTS_CLASSES_H
#define TEST_INTEROP_CXX_NAMESPACE_INPUTS_CLASSES_H

namespace ClassesNS1 {
struct BasicStruct {
  const char *basicMember() __attribute__((language_attr("import_unsafe"))) {
    return "ClassesNS1::BasicStruct::basicMember";
  }
};
struct ForwardDeclaredStruct;
} // namespace ClassesNS1

struct ClassesNS1::ForwardDeclaredStruct {
  const char *basicMember() __attribute__((language_attr("import_unsafe"))) {
    return "ClassesNS1::ForwardDeclaredStruct::basicMember";
  }
};

namespace ClassesNS1 {
namespace ClassesNS2 {
struct BasicStruct {
  const char *basicMember() __attribute__((language_attr("import_unsafe"))) {
    return "ClassesNS1::ClassesNS2::BasicStruct::basicMember";
  }
};
struct ForwardDeclaredStruct;
struct DefinedInDefs;
} // namespace ClassesNS2
} // namespace ClassesNS1

namespace ClassesNS1 {
struct ClassesNS2::ForwardDeclaredStruct {
  const char *basicMember() __attribute__((language_attr("import_unsafe"))) {
    return "ClassesNS1::ClassesNS2::ForwardDeclaredStruct::basicMember";
  }
};
} // namespace ClassesNS1

namespace ClassesNS3 {
struct BasicStruct {
  const char *basicMember() __attribute__((language_attr("import_unsafe"))) {
    return "ClassesNS3::BasicStruct::basicMember";
  }
};
} // namespace ClassesNS3

namespace GlobalAliasToNS1 = ClassesNS1;

namespace ClassesNS4 {
namespace AliasToGlobalNS1 = ::ClassesNS1;
namespace AliasToGlobalNS2 = ::ClassesNS1::ClassesNS2;

namespace ClassesNS5 {
struct BasicStruct {};
} // namespace ClassesNS5

namespace AliasToInnerNS5 = ClassesNS5;
namespace AliasToNS2 = ClassesNS1::ClassesNS2;

namespace AliasChainToNS1 = GlobalAliasToNS1;
namespace AliasChainToNS2 = AliasChainToNS1::ClassesNS2;
} // namespace ClassesNS4

namespace ClassesNS5 {
struct BasicStruct {};
namespace AliasToAnotherNS5 = ClassesNS4::ClassesNS5;

namespace ClassesNS5 {
struct BasicStruct {};
namespace AliasToNS5NS5 = ClassesNS5;
} // namespace ClassesNS5

namespace AliasToGlobalNS5 = ::ClassesNS5;
namespace AliasToLocalNS5 = ClassesNS5;
namespace AliasToNS5 = ::ClassesNS5::ClassesNS5;
} // namespace ClassesNS5

#endif // TEST_INTEROP_CXX_NAMESPACE_INPUTS_CLASSES_H
