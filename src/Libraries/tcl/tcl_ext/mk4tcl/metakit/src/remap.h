/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 30, 2022.
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

// remap.h --
// $Id: remap.h 1230 2007-03-09 15:58:53Z jcw $
// This is part of Metakit, see http://www.equi4.com/metakit.html

/** @file
 * Encapsulation of the (re)mapping viewers
 */

#ifndef __REMAP_H__
#define __REMAP_H__

/////////////////////////////////////////////////////////////////////////////
// Declarations in this file

class c4_Sequence; // not defined here

extern c4_CustomViewer *f4_CreateReadOnly(c4_Sequence &);
extern c4_CustomViewer *f4_CreateHash(c4_Sequence &, int, c4_Sequence * = 0);
extern c4_CustomViewer *f4_CreateBlocked(c4_Sequence &);
extern c4_CustomViewer *f4_CreateOrdered(c4_Sequence &, int);
extern c4_CustomViewer *f4_CreateIndexed(c4_Sequence &, c4_Sequence &, const
  c4_View &, bool = false);

/////////////////////////////////////////////////////////////////////////////

#endif
