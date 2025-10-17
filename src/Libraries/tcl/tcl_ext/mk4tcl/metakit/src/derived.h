/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 12, 2024.
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

// derived.h --
// $Id: derived.h 1230 2007-03-09 15:58:53Z jcw $
// This is part of Metakit, the homepage is http://www.equi4.com/metakit.html

/** @file
 * Encapsulation of derived view classes
 */

#ifndef __DERIVED_H__
#define __DERIVED_H__

/////////////////////////////////////////////////////////////////////////////
// Declarations in this file

class c4_Cursor; // not defined here
class c4_Sequence; // not defined here

extern c4_Sequence *f4_CreateFilter(c4_Sequence &, c4_Cursor, c4_Cursor);
extern c4_Sequence *f4_CreateSort(c4_Sequence &, c4_Sequence * = 0);
extern c4_Sequence *f4_CreateProject(c4_Sequence &, c4_Sequence &, bool,
  c4_Sequence * = 0);

/////////////////////////////////////////////////////////////////////////////

#endif
