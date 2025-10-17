/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 21, 2025.
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

#ifndef ARRAY2_H
#define ARRAY2_H

#include "Array1.h"
#include <stdexcept>
#include <string>

class Array2
{
public:

  // Default constructor
  Array2();

  // Size/array constructor
  Array2(int nrows, int ncols, long* data=0);

  // Copy constructor
  Array2(const Array2 & source);

  // Destructor
  ~Array2();

  // Assignment operator
  Array2 & operator=(const Array2 & source);

  // Equals operator
  bool operator==(const Array2 & other) const;

  // Length accessors
  int nrows() const;
  int ncols() const;

  // Resize array
  void resize(int nrows, int ncols, long* data=0);

  // Set item accessor
  Array1 & operator[](int i);

  // Get item accessor
  const Array1 & operator[](int i) const;

  // String output
  std::string asString() const;

  // Get view
  void view(int* nrows, int* ncols, long** data) const;

private:
  // Members
  bool _ownData;
  int _nrows;
  int _ncols;
  long * _buffer;
  Array1 * _rows;

  // Methods
  void allocateMemory();
  void allocateRows();
  void deallocateMemory();
};

#endif
