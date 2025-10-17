/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 3, 2021.
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
trace_SQLDrivers (int trace_leave, int retcode,
  SQLHENV		  EnvironmentHandle,
  SQLUSMALLINT		  Direction,
  SQLCHAR		* DriverDescription,
  SQLSMALLINT		  BufferLength1,
  SQLSMALLINT		* DescriptionLengthPtr,
  SQLCHAR		* DriverAttributes,
  SQLSMALLINT		  BufferLength2,
  SQLSMALLINT		* AttributesLengthPtr)
{
  /* Trace function */
  _trace_print_function (en_Drivers, trace_leave, retcode);

  /* Trace Arguments */
  _trace_handle (SQL_HANDLE_ENV, EnvironmentHandle);
  _trace_direction (Direction);
  _trace_string (DriverDescription, BufferLength1, DescriptionLengthPtr,
      TRACE_OUTPUT_SUCCESS);
  _trace_stringlen ("SQLSMALLINT", BufferLength1);
  _trace_smallint_p (DescriptionLengthPtr, TRACE_OUTPUT_SUCCESS);
  _trace_string (DriverAttributes, BufferLength2, AttributesLengthPtr,
      TRACE_OUTPUT_SUCCESS);
  _trace_stringlen ("SQLSMALLINT", BufferLength2);
  _trace_smallint_p (AttributesLengthPtr, TRACE_OUTPUT_SUCCESS);
}


#if ODBCVER >= 0x0300
void
trace_SQLDriversW (int trace_leave, int retcode,
  SQLHENV		  EnvironmentHandle,
  SQLUSMALLINT		  Direction,
  SQLWCHAR		* DriverDescription,
  SQLSMALLINT		  BufferLength1,
  SQLSMALLINT		* DescriptionLengthPtr,
  SQLWCHAR		* DriverAttributes,
  SQLSMALLINT		  BufferLength2,
  SQLSMALLINT		* AttributesLengthPtr)
{
  /* Trace function */
  _trace_print_function (en_DriversW, trace_leave, retcode);

  /* Trace Arguments */
  _trace_handle (SQL_HANDLE_ENV, EnvironmentHandle);
  _trace_direction (Direction);
  _trace_string_w (DriverDescription, BufferLength1, DescriptionLengthPtr,
      TRACE_OUTPUT_SUCCESS);
  _trace_stringlen ("SQLSMALLINT", BufferLength1);
  _trace_smallint_p (DescriptionLengthPtr, TRACE_OUTPUT_SUCCESS);
  _trace_string_w (DriverAttributes, BufferLength2, AttributesLengthPtr,
      TRACE_OUTPUT_SUCCESS);
  _trace_stringlen ("SQLSMALLINT", BufferLength2);
  _trace_smallint_p (AttributesLengthPtr, TRACE_OUTPUT_SUCCESS);
}
#endif
