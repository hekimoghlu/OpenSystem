/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 7, 2024.
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

#if HAVE_CONFIG_H
#include <config.h>
#endif


#include <stdio.h>
#include <string.h>
#include <dce/rpc.h>
#include <dce/dce_error.h>
#include <stdlib.h>
#include "misc.h"

void
chk_dce_err(ecode, where, why, fatal)
     error_status_t ecode;
     const char * where;
     const char * why;
     unsigned int fatal;
{

  dce_error_string_t errstr;
  int error_status;

  if (ecode != error_status_ok)
    {
       dce_error_inq_text(ecode, errstr, &error_status);
       if (error_status == error_status_ok)
	 printf("ERROR.  where = <%s> why = <%s> error code = 0x%x"
		"reason = <%s>\n",
	      where, why, ecode, errstr);
       else
	 printf("ERROR.  where = <%s> why = <%s> error code = 0x%x\n",
	      where, why, ecode);

       if (fatal) exit(1);
    }
}

void* midl_user_allocate(idl_size_t size)
{
    void *result = malloc(size);
    fprintf(stderr, "USER_ALLOC: %u -> %p\n", size, result);
    return result;
}

void midl_user_free(void* obj)
{
    fprintf(stderr, "USER_FREE: %p\n", obj);
}
