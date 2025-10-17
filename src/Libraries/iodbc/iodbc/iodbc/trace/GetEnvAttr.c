/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 1, 2022.
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


#if ODBCVER >= 0x0300
void
_trace_envattr_type (SQLINTEGER type)
{
  char *ptr = "unknown environment attribute";

  switch (type)
    {
      _S (SQL_ATTR_CONNECTION_POOLING);
      _S (SQL_ATTR_CP_MATCH);
      _S (SQL_ATTR_ODBC_VERSION);
      _S (SQL_ATTR_OUTPUT_NTS);
    }

  trace_emit ("\t\t%-15.15s   %d (%s)\n", "SQLINTEGER ", (int) type, ptr);
}


void
trace_SQLGetEnvAttr (int trace_leave, int retcode,
  SQLHENV		  EnvironmentHandle,
  SQLINTEGER		  Attribute,
  SQLPOINTER		  ValuePtr,
  SQLINTEGER	 	  BufferLength,
  SQLINTEGER		* StringLengthPtr)
{
  /* Trace function */
  _trace_print_function (en_GetEnvAttr, trace_leave, retcode);

  /* Trace Arguments */
  _trace_handle (SQL_HANDLE_ENV, EnvironmentHandle);
  _trace_envattr_type (Attribute);
  _trace_pointer (ValuePtr);
  _trace_bufferlen (BufferLength);
  _trace_integer_p (StringLengthPtr, TRACE_OUTPUT_SUCCESS);
}
#endif
