/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 2, 2023.
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
/*
 * Mach Operating System
 * Copyright (c) 1991,1990 Carnegie Mellon University
 * All Rights Reserved.
 *
 * Permission to use, copy, modify and distribute this software and its
 * documentation is hereby granted, provided that both the copyright
 * notice and this permission notice appear in all copies of the
 * software, derivative works or modified versions, and any portions
 * thereof, and that both notices appear in supporting documentation.
 *
 * CARNEGIE MELLON ALLOWS FREE USE OF THIS SOFTWARE IN ITS
 * CONDITION.  CARNEGIE MELLON DISCLAIMS ANY LIABILITY OF ANY KIND FOR
 * ANY DAMAGES WHATSOEVER RESULTING FROM THE USE OF THIS SOFTWARE.
 *
 * Carnegie Mellon requests users of this software to return to
 *
 *  Software Distribution Coordinator  or  Software.Distribution@CS.CMU.EDU
 *  School of Computer Science
 *  Carnegie Mellon University
 *  Pittsburgh PA 15213-3890
 *
 * any improvements or extensions that they make and grant Carnegie the
 * rights to redistribute these changes.
 */
/*
 * HISTORY
 * 07-Apr-89  Richard Draves (rpd) at Carnegie-Mellon University
 *  Extensive revamping.  Added polymorphic arguments.
 *  Allow multiple variable-sized inline arguments in messages.
 *
 * 27-May-87  Richard Draves (rpd) at Carnegie-Mellon University
 *  Created.
 */

#ifndef _STATEMENT_H
#define _STATEMENT_H

#include "routine.h"

typedef enum statement_kind
{
  skRoutine,
  skImport,
  skUImport,
  skSImport,
  skDImport,
  skIImport,
  skRCSDecl
} statement_kind_t;

typedef struct statement
{
  statement_kind_t stKind;
  struct statement *stNext;
  union
  {
    /* when stKind == skRoutine */
    routine_t *_stRoutine;
    /* when stKind == skImport, skUImport, skSImport, skDImport, skIImport */
    string_t _stFileName;
  } data;
} statement_t;

#define stRoutine data._stRoutine
#define stFileName  data._stFileName

#define stNULL  ((statement_t *) 0)

/* stNext will be initialized to put the statement in the list */
extern statement_t *stAlloc(void);

/* list of statements, in order they occur in the .defs file */
extern statement_t *stats;

#endif  /* _STATEMENT_H */
