/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 24, 2025.
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
//  utils.c
//  dcerpctest_server
//
//  Created by William Conway on 12/1/23.
//

#if HAVE_CONFIG_H
#include <config.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <compat/dcerpc.h>
#include "utils.h"

void
chk_dce_err(error_status_t ecode, const char *routine, const char *ctx, unsigned int fatal)
{
  dce_error_string_t errstr;
  int error_status;

  if (ecode != error_status_ok)
    {
       dce_error_inq_text(ecode, errstr, &error_status);
        if (error_status == error_status_ok) {
            printf("chk_dce_err: routine: %s, ctx: %s, rpc_code = 0x%x, error_desc: %s\n",
                   routine ? routine : "n/a", ctx ? ctx : "n/a", ecode, errstr);
        } else
            printf("chk_dce_err: routine: %s ctx: %s rpc_code = 0x%x, error_desc: <not available>\n",
                   routine ? routine : "n/a", ctx ? ctx : "n/a", ecode);

        if (fatal) {
            printf("chk_dce_err: exiting on fatal error\n");
            exit(1);
        }
    }
}

