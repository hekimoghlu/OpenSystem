/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 27, 2022.
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

struct HasPrivatePointerField {
private:
  const int *ptr;
};

struct HasProtectedPointerField {
protected:
  const int *ptr;
};

struct HasPublicPointerField {
  const int *ptr;
};

struct HasPrivateNonSendableField {
private:
  HasPrivatePointerField f;
};

struct HasProtectedNonSendableField {
protected:
  HasProtectedPointerField f;
};

struct HasPublicNonSendableField {
  HasPublicPointerField f;
};

struct DerivedFromHasPublicPointerField : HasPublicPointerField {};
struct DerivedFromHasPublicNonSendableField : HasPublicNonSendableField {};
struct DerivedFromHasPrivatePointerField : HasPrivatePointerField {};

struct DerivedPrivatelyFromHasPublicPointerField : private HasPublicPointerField {};
struct DerivedPrivatelyFromHasPublicNonSendableField : private HasPublicNonSendableField {};
struct DerivedPrivatelyFromHasPrivatePointerField : private HasPrivatePointerField {};

struct DerivedProtectedFromHasPublicPointerField : protected HasPublicPointerField {};
struct DerivedProtectedFromHasPublicNonSendableField : protected HasPublicNonSendableField {};
struct DerivedProtectedFromHasPrivatePointerField : protected HasPrivatePointerField {};
