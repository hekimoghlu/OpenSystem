/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 5, 2025.
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

#ifndef TEST_INTEROP_CXX_CLASS_INPUTS_MUTABILITY_ANNOTATIONS_H
#define TEST_INTEROP_CXX_CLASS_INPUTS_MUTABILITY_ANNOTATIONS_H

struct HasConstMethodAnnotatedAsMutating {
  int a;

  int annotatedMutating() const __attribute__((__language_attr__("mutating"))) {
    const_cast<HasConstMethodAnnotatedAsMutating *>(this)->a++;
    return a;
  }

  int annotatedMutatingWithOtherAttrs() const __attribute__((__language_attr__("public"))) __attribute__((__language_attr__("mutating"))) {
    const_cast<HasConstMethodAnnotatedAsMutating *>(this)->a++;
    return a;
  }
};

struct HasMutableProperty {
  mutable int a;
  int b;

  int annotatedNonMutating() const __attribute__((__language_attr__("nonmutating"))) {
    return b;
  }

  int noAnnotation() const { return b; }

  // expected-warning@+1 {{attribute 'mutating' is ignored when combined with attribute 'nonmutating'}}
  int contradictingAnnotations() const __attribute__((__language_attr__("nonmutating"))) __attribute__((__language_attr__("mutating"))) {
    return b;
  }

  int duplicateAnnotations() const __attribute__((__language_attr__("nonmutating"))) __attribute__((__language_attr__("nonmutating"))) {
    return b;
  }
};

struct NoMutableProperty {
  int a;

  // expected-warning@+1 {{attribute 'nonmutating' has no effect without any mutable fields}}
  int isConst() const __attribute__((__language_attr__("nonmutating"))) {
    return a;
  }

  // expected-warning@+2 {{attribute 'nonmutating' has no effect without any mutable fields}}
  // expected-warning@+1 {{attribute 'nonmutating' has no effect on non-const method}}
  int nonConst() __attribute__((__language_attr__("nonmutating"))) { return a; }
};

#endif // TEST_INTEROP_CXX_CLASS_INPUTS_MUTABILITY_ANNOTATIONS_H
