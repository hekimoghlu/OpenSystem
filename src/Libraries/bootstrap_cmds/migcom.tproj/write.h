/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 9, 2023.
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
 * Extensive revamping.  Added polymorphic arguments.
 * Allow multiple variable-sized inline arguments in messages.
 *
 * 27-May-87  Richard Draves (rpd) at Carnegie-Mellon University
 * Created.
 */

#ifndef _WRITE_H
#define _WRITE_H

#include <stdio.h>
#include "statement.h"

extern void WriteUserHeader( FILE *file, statement_t *stats );
extern void WriteServerHeader( FILE *file, statement_t *stats );
extern void WriteServerRoutine( FILE *file, routine_t *rt );
extern void WriteInternalHeader( FILE *file, statement_t *stats );
extern void WriteDefinesHeader( FILE *file, statement_t *stats );
extern void WriteUser( FILE *file, statement_t *stats );
extern void WriteUserIndividual( statement_t *stats );
extern void WriteServer( FILE *file, statement_t *stats );
extern void WriteIncludes( FILE *file, boolean_t isuser, boolean_t is_def );
extern void WriteImplImports( FILE *file, statement_t *stats, boolean_t isuser );
extern void WriteApplDefaults( FILE *file, char *dir );
extern void WriteApplMacro( FILE *file, char *dir, char *when, routine_t *rt );
extern void WriteBogusServerRoutineAnnotationDefine( FILE *file );

#endif /* _WRITE_H */
