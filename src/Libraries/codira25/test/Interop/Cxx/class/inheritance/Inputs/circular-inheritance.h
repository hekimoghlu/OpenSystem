/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 12, 2023.
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

#ifndef CIRCULAR_INHERITANCE_H
#define CIRCULAR_INHERITANCE_H

struct DinoEgg {
  void dinoEgg(void) const {}
};

template <typename Chicken>
struct Egg;

template <>
struct Egg<void> : DinoEgg {
  Egg() {}
  void voidEgg(void) const {}
};

template <typename Chicken>
struct Egg : Egg<void> {
  Egg(Chicken _chicken) {}
  Chicken chickenEgg(Chicken c) const { return c; }
};

typedef Egg<void> VoidEgg;
typedef Egg<bool> BoolEgg;
typedef Egg<Egg<void>> EggEgg;

struct NewEgg : Egg<int> {
  NewEgg() : Egg<int>(555) {}
  void newEgg() const {}
};

#endif // !CIRCULAR_INHERITANCE_H
