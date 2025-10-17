/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 16, 2023.
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

//
// Array class for the CUPS PPD Compiler.
//
// Copyright 2007-2019 by Apple Inc.
// Copyright 2002-2005 by Easy Software Products.
//
// Licensed under Apache License v2.0.  See the file "LICENSE" for more information.
//

//
// Include necessary headers...
//

#include "ppdc-private.h"


//
// 'ppdcArray::ppdcArray()' - Create a new array.
//

ppdcArray::ppdcArray(ppdcArray *a)
  : ppdcShared()
{
  PPDC_NEW;

  if (a)
  {
    count = a->count;
    alloc = count;

    if (count)
    {
      // Make a copy of the array...
      data = new ppdcShared *[count];

      memcpy(data, a->data, (size_t)count * sizeof(ppdcShared *));

      for (size_t i = 0; i < count; i ++)
        data[i]->retain();
    }
    else
      data = 0;
  }
  else
  {
    count = 0;
    alloc = 0;
    data  = 0;
  }

  current = 0;
}


//
// 'ppdcArray::~ppdcArray()' - Destroy an array.
//

ppdcArray::~ppdcArray()
{
  PPDC_DELETE;

  for (size_t i = 0; i < count; i ++)
    data[i]->release();

  if (alloc)
    delete[] data;
}


//
// 'ppdcArray::add()' - Add an element to an array.
//

void
ppdcArray::add(ppdcShared *d)
{
  ppdcShared	**temp;


  if (count >= alloc)
  {
    alloc += 10;
    temp  = new ppdcShared *[alloc];

    memcpy(temp, data, (size_t)count * sizeof(ppdcShared *));

    delete[] data;
    data = temp;
  }

  data[count++] = d;
}


//
// 'ppdcArray::first()' - Return the first element in the array.
//

ppdcShared *
ppdcArray::first()
{
  current = 0;

  if (current >= count)
    return (0);
  else
    return (data[current ++]);
}


//
// 'ppdcArray::next()' - Return the next element in the array.
//

ppdcShared *
ppdcArray::next()
{
  if (current >= count)
    return (0);
  else
    return (data[current ++]);
}


//
// 'ppdcArray::remove()' - Remove an element from the array.
//

void
ppdcArray::remove(ppdcShared *d)		// I - Data element
{
  size_t	i;				// Looping var


  for (i = 0; i < count; i ++)
    if (d == data[i])
      break;

  if (i >= count)
    return;

  count --;
  d->release();

  if (i < count)
    memmove(data + i, data + i + 1, (size_t)(count - i) * sizeof(ppdcShared *));
}
