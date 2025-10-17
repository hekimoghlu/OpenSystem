/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 28, 2024.
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
trace_SQLSetEnvAttr (int trace_leave, int retcode,
  SQLHENV		  EnvironmentHandle,
  SQLINTEGER		  Attribute,
  SQLPOINTER		  ValuePtr,
  SQLINTEGER		  StringLength)
{
  /* Trace function */
  _trace_print_function (en_SetEnvAttr, trace_leave, retcode);

  /* Trace Arguments */
  _trace_handle (SQL_HANDLE_ENV, EnvironmentHandle);
  _trace_envattr_type (Attribute);
  _trace_pointer (ValuePtr);
  _trace_bufferlen (StringLength);
}
#endif
