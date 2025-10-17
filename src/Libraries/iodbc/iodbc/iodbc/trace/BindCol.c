/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 4, 2023.
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
trace_SQLBindCol (int trace_leave, int retcode,
    SQLHSTMT		  StatementHandle,
    SQLUSMALLINT	  ColumnNumber,
    SQLSMALLINT		  TargetType,
    SQLPOINTER		  TargetValuePtr,
    SQLLEN		  BufferLength,
    SQLLEN		* Strlen_or_IndPtr)
{
  /* Trace function */
  _trace_print_function (en_BindCol, trace_leave, retcode);

  /* Trace Arguments */
  _trace_handle (SQL_HANDLE_STMT, StatementHandle);
  _trace_usmallint (ColumnNumber);
  _trace_c_type (TargetType);
  _trace_pointer (TargetValuePtr);
  _trace_len (BufferLength);
  _trace_len_p (Strlen_or_IndPtr, TRACE_OUTPUT_SUCCESS);
}
