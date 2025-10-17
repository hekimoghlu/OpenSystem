/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 7, 2022.
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

#ifndef TEST_INTEROP_CXX_NAMESPACE_INPUTS_FREE_FUNCTION_H
#define TEST_INTEROP_CXX_NAMESPACE_INPUTS_FREE_FUNCTION_H

namespace FunctionsNS1 {
inline const char *basicFunctionTopLevel() {
  return "FunctionsNS1::basicFunctionTopLevel";
}
inline const char *forwardDeclared();
inline const char *definedOutOfLine();

struct X {};
inline const char *operator+(X, X) { return "FunctionsNS1::operator+(X, X)"; }
} // namespace FunctionsNS1

namespace FunctionsNS1 {
inline const char *forwardDeclared() { return "FunctionsNS1::forwardDeclared"; }
} // namespace FunctionsNS1

inline const char *FunctionsNS1::definedOutOfLine() {
  return "FunctionsNS1::definedOutOfLine";
}

namespace FunctionsNS1 {
namespace FunctionsNS2 {
inline const char *basicFunctionSecondLevel() {
  return "FunctionsNS1::FunctionsNS2::basicFunctionSecondLevel";
}
} // namespace FunctionsNS2
} // namespace FunctionsNS1

namespace FunctionsNS1 {
namespace FunctionsNS2 {
namespace FunctionsNS3 {
inline const char *basicFunctionLowestLevel() {
  return "FunctionsNS1::FunctionsNS2::FunctionsNS3::basicFunctionLowestLevel";
}
} // namespace FunctionsNS3
} // namespace FunctionsNS2
} // namespace FunctionsNS1

namespace FunctionsNS1 {
inline const char *definedInDefs();
}

namespace FunctionsNS1 {
inline const char *sameNameInChild() { return "FunctionsNS1::sameNameInChild"; }
inline const char *sameNameInSibling() {
  return "FunctionsNS1::sameNameInSibling";
}
namespace FunctionsNS2 {
inline const char *sameNameInChild() {
  return "FunctionsNS1::FunctionsNS2::sameNameInChild";
}
} // namespace FunctionsNS2
} // namespace FunctionsNS1

namespace FunctionsNS4 {
inline const char *sameNameInSibling() {
  return "FunctionsNS4::sameNameInSibling";
}
} // namespace FunctionsNS4

namespace FunctionsNS1 {
namespace FunctionsNS2 {
namespace FunctionsNS3 {
struct Y {};
inline bool operator==(Y, Y) { return true; }
} // namespace FunctionsNS3
} // namespace FunctionsNS2
} // namespace FunctionsNS1

#endif // TEST_INTEROP_CXX_NAMESPACE_INPUTS_FREE_FUNCTION_H
