/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 20, 2023.
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

#ifndef ARRAY1_H
#define ARRAY1_H

#include <stdexcept>
#include <string>

class Array1
{
public:

  // Default/length/array constructor
  Array1(int length = 0, long* data = 0);

  // Copy constructor
  Array1(const Array1 & source);

  // Destructor
  ~Array1();

  // Assignment operator
  Array1 & operator=(const Array1 & source);

  // Equals operator
  bool operator==(const Array1 & other) const;

  // Length accessor
  int length() const;

  // Resize array
  void resize(int length, long* data = 0);

  // Set item accessor
  long & operator[](int i);

  // Get item accessor
  const long & operator[](int i) const;

  // String output
  std::string asString() const;

  // Get view
  void view(long** data, int* length) const;

private:
  // Members
  bool _ownData;
  int _length;
  long * _buffer;

  // Methods
  void allocateMemory();
  void deallocateMemory();
};

#endif
