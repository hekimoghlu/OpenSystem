/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 19, 2023.
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
**
**  NAME
**
**      sec_id_pickle.h
**
**  FACILITY:
**
**      Remote Procedure Call (RPC)
**
**  ABSTRACT:
**
**      Types and routines for PAC management.
*/
#ifndef _SEC_ID_PICKLE_H
#define _SEC_ID_PICKLE_H  1

#include <commonp.h>
#include <com.h>
#include <comp.h>

#include <dce/id_base.h>
#include <dce/sec_authn.h>

typedef struct pickle_handle_s * pickle_handle_t;

/*
 * Functions
 */

/* s e c _ p i c k l e _ c r e a t e
 *
 * Create a pickling context.  This must be called to obtain a pickling
 * context before any pickling calls can be performed.
 */
pickle_handle_t sec_pickle_create ( void );

/* s e c _ p i c k l e _ r e l e a s e
 *
 * Terminate a pickling context.  This function will release any storage
 * associated with the pickling context.
 */
void sec_pickle_release ( pickle_handle_t * /*p*/);

/* s e c _ i d _ p a c _ f r e e
 *
 * Release dynamic storage associated with a PAC.
 */

void sec_id_pac_free ( sec_id_pac_t *) ;

/* s e c _ i d _ p a c _ p i c k l e
 *
 * Pickle a pac.
 */
extern void     sec_id_pac_pickle (
        /* [in] */      pickle_handle_t          /*pickle_handle*/,
        /* [in] */      sec_id_pac_t            *  /*pac*/,
        /* [out] */     sec_id_pickled_pac_t    **  /*pickled_pac*/
  );

/* s e c _ i d _ p a c _ u n p i c k l e
 *
 * unpickle a pac
 */

extern void     sec_id_pac_unpickle (
        /* [in] */      sec_id_pickled_pac_t    *  /*pickled_pac*/,
        /* [out] */     sec_id_pac_t            *  /*pac*/
  );
#endif
