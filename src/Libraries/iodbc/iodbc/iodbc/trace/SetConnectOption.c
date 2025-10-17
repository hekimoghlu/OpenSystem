/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 16, 2025.
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
#include "trace.h"


void
trace_SQLSetConnectOption (int trace_leave, int retcode,
  SQLHDBC		  hdbc,
  SQLUSMALLINT		  fOption,
  SQLULEN	  vParam)
{
  /* Trace function */
  _trace_print_function (en_SetConnectOption, trace_leave, retcode);

  /* Trace Arguments */
  _trace_handle (SQL_HANDLE_DBC, hdbc);
  _trace_connopt_type (fOption);
  _trace_ulen (vParam);
}


#if ODBCVER >= 0x0300
void
trace_SQLSetConnectOptionW (int trace_leave, int retcode,
  SQLHDBC		  hdbc,
  SQLUSMALLINT		  fOption,
  SQLULEN		  vParam)
{
  /* Trace function */
  _trace_print_function (en_SetConnectOptionW, trace_leave, retcode);

  /* Trace Arguments */
  _trace_handle (SQL_HANDLE_DBC, hdbc);
  _trace_connopt_type (fOption);
  _trace_ulen (vParam);
}
#endif
