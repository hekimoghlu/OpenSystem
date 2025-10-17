/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 20, 2024.
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
/* This header file is private to the uuconf memory allocation
   routines, and should not be included by any other files.  */

/* We want to be able to keep track of allocated memory blocks, so
   that we can free them up later.  This will let us free up all the
   memory allocated to hold information for a system, for example.  We
   do this by allocating large chunks and doling them out.  Calling
   uuconf_malloc_block will return a pointer to a magic cookie which
   can then be passed to uuconf_malloc and uuconf_free.  Passing the
   pointer to uuconf_free_block will free all memory allocated for
   that block.  */

/* We allocate this much space in each block.  On most systems, this
   will make the actual structure 1024 bytes, which may be convenient
   for some types of memory allocators.  */
#define CALLOC_SIZE (1008)

/* This is the actual structure of a block.  */
struct sblock
{
  /* Next block in linked list.  */
  struct sblock *qnext;
  /* Index of next free spot.  */
  size_t ifree;
  /* Last value returned by uuconf_malloc for this block.  */
  pointer plast;
  /* List of additional memory blocks.  */
  struct sadded *qadded;
  /* Buffer of data.  We put it in a union with a double to make sure
     it is adequately aligned.  */
  union
    {
      char ab[CALLOC_SIZE];
      double l;
    } u;
};

/* There is a linked list of additional memory blocks inserted by
   uuconf_add_block.  */
struct sadded
{
  /* The next in the list.  */
  struct sadded *qnext;
  /* The added block.  */
  pointer padded;
};
