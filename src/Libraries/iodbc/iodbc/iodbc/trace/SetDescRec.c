/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 2, 2024.
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
trace_SQLSetDescRec (int trace_leave, int retcode,
  SQLHDESC		  DescriptorHandle,
  SQLSMALLINT		  RecNumber,
  SQLSMALLINT		  Type,
  SQLSMALLINT		  SubType,
  SQLLEN		  Length,
  SQLSMALLINT		  Precision,
  SQLSMALLINT		  Scale,
  SQLPOINTER		  Data,
  SQLLEN		* StringLength,
  SQLLEN		* Indicator)
{
  /* Trace function */
  _trace_print_function (en_SetDescRec, trace_leave, retcode);

  /* Trace Arguments */
  _trace_handle (SQL_HANDLE_DESC, DescriptorHandle);
  _trace_smallint (RecNumber);
  _trace_smallint (Type);
  _trace_smallint (SubType);
  _trace_len (Length);
  _trace_smallint (Precision);
  _trace_smallint (Scale);
  _trace_pointer (Data);
  _trace_len_p (StringLength, TRACE_OUTPUT_SUCCESS);
  _trace_len_p (Indicator, TRACE_OUTPUT_SUCCESS);
}
#endif
