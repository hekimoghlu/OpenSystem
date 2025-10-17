/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 3, 2025.
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
#include <iodbc.h>
#include <odbcinst.h>

#include "iodbc_error.h"
#include "misc.h"


BOOL INSTAPI
SQLGetConfigMode (UWORD * pwConfigMode)
{
  BOOL retcode = FALSE;

  /* Clear errors */
  CLEAR_ERROR ();

  /* Check input parameter */
  if (!pwConfigMode)
    {
      PUSH_ERROR (ODBC_ERROR_OUT_OF_MEM);
    }
  else
    {
      *pwConfigMode = configMode;
      retcode = TRUE;
    }

  return retcode;
}
