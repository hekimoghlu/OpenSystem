/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 24, 2021.
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

#ifndef TEST_INTEROP_CXX_NAMESPACE_INPUTS_TEMPLATES_WITH_FORWARD_DECL_H
#define TEST_INTEROP_CXX_NAMESPACE_INPUTS_TEMPLATES_WITH_FORWARD_DECL_H

namespace NS1 {

template <typename T>
struct Decl;

typedef Decl<int> di;

} // namespace NS1

namespace NS1 {

template <typename T>
struct ForwardDeclared;

template <typename T>
struct Decl {
  typedef T MyInt;
  ForwardDeclared<T> fwd;
  const static MyInt intValue = -1;
};

template <typename T>
const typename Decl<T>::MyInt Decl<T>::intValue;

} // namespace NS1

namespace NS1 {

template <typename T>
struct ForwardDeclared {};

} // namespace NS1

#endif // TEST_INTEROP_CXX_NAMESPACE_INPUTS_TEMPLATES_WITH_FORWARD_DECL_H
