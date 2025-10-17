/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 27, 2024.
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
 * Copyright (c) 1987 Carnegie-Mellon University
 * All rights reserved.  The CMU software License Agreement specifies
 * the terms and conditions for use and redistribution.
 */
/*
 * Mach Interface Generator errors
 *
 * $Header: /Users/Shared/bootstrap_cmds/bootstrap_cmds/migcom.tproj/mig_errors.h,v 1.1.1.2 2000/01/11 00:36:18 wsanchez Exp $
 *
 * HISTORY
 * 07-Apr-89  Richard Draves (rpd) at Carnegie-Mellon University
 *  Extensive revamping.  Added polymorphic arguments.
 *  Allow multiple variable-sized inline arguments in messages.
 *
 * 28-Apr-88  Bennet Yee (bsy) at Carnegie-Mellon University
 *  Put mig_symtab back.
 *
 *  2-Dec-87  David Golub (dbg) at Carnegie-Mellon University
 *  Added MIG_ARRAY_TOO_LARGE.
 *
 * 25-May-87  Richard Draves (rpd) at Carnegie-Mellon University
 *  Added definition of death_pill_t.
 *
 * 31-Jul-86  Michael Young (mwyoung) at Carnegie-Mellon University
 *  Created.
 */
#ifndef _MIG_ERRORS_H
#define _MIG_ERRORS_H

#include <mach/kern_return.h>
#include <mach/message.h>

#define MIG_TYPE_ERROR      -300    /* Type check failure */
#define MIG_REPLY_MISMATCH  -301    /* Wrong return message ID */
#define MIG_REMOTE_ERROR    -302    /* Server detected error */
#define MIG_BAD_ID          -303    /* Bad message ID */
#define MIG_BAD_ARGUMENTS   -304    /* Server found wrong arguments */
#define MIG_NO_REPLY        -305    /* Server shouldn't reply */
#define MIG_EXCEPTION       -306    /* Server raised exception */
#define MIG_ARRAY_TOO_LARGE -307    /* User specified array not large enough
                                       to hold returned array */

typedef struct {
  msg_header_t  Head;
  msg_type_t    RetCodeType;
  kern_return_t RetCode;
} death_pill_t;

typedef struct mig_symtab {
  char  *ms_routine_name;
  int   ms_routine_number;
#ifdef hc
  void
#else
  int
#endif
        (*ms_routine)();
} mig_symtab_t;

#endif  _MIG_ERRORS_H
