/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 7, 2024.
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
_trace_connattr_type (SQLINTEGER type)
{
  char *ptr = "unknown connection attribute";

  switch (type)
    {
      _S (SQL_ATTR_ACCESS_MODE);
      _S (SQL_ATTR_AUTOCOMMIT);
      _S (SQL_ATTR_AUTO_IPD);
      _S (SQL_ATTR_CONNECTION_DEAD);
      _S (SQL_ATTR_CONNECTION_TIMEOUT);
      _S (SQL_ATTR_CURRENT_CATALOG);
      _S (SQL_ATTR_DISCONNECT_BEHAVIOR);
      _S (SQL_ATTR_ENLIST_IN_DTC);
      _S (SQL_ATTR_ENLIST_IN_XA);
      _S (SQL_ATTR_LOGIN_TIMEOUT);
      _S (SQL_ATTR_METADATA_ID);
      _S (SQL_ATTR_ODBC_CURSORS);
      _S (SQL_ATTR_PACKET_SIZE);
      _S (SQL_ATTR_QUIET_MODE);
      _S (SQL_ATTR_TRACE);
      _S (SQL_ATTR_TRACEFILE);
      _S (SQL_ATTR_TRANSLATE_LIB);
      _S (SQL_ATTR_TRANSLATE_OPTION);
      _S (SQL_ATTR_TXN_ISOLATION);
    }

  trace_emit ("\t\t%-15.15s   %d (%s)\n", "SQLINTEGER ", (int) type, ptr);
}


void
trace_SQLGetConnectAttr (int trace_leave, int retcode,
  SQLHDBC		  ConnectionHandle,
  SQLINTEGER		  Attribute,
  SQLPOINTER		  ValuePtr,
  SQLINTEGER		  BufferLength,
  SQLINTEGER		* StringLengthPtr)
{
  /* Trace function */
  _trace_print_function (en_GetConnectAttr, trace_leave, retcode);

  /* Trace Arguments */
  _trace_handle (SQL_HANDLE_DBC, ConnectionHandle);
  _trace_connattr_type (Attribute);
  _trace_pointer (ValuePtr);
  _trace_bufferlen (BufferLength);
  _trace_integer_p (StringLengthPtr, TRACE_OUTPUT_SUCCESS);
}


void
trace_SQLGetConnectAttrW (int trace_leave, int retcode,
  SQLHDBC		  ConnectionHandle,
  SQLINTEGER		  Attribute,
  SQLPOINTER		  ValuePtr,
  SQLINTEGER		  BufferLength,
  SQLINTEGER		* StringLengthPtr)
{
  /* Trace function */
  _trace_print_function (en_GetConnectAttrW, trace_leave, retcode);

  /* Trace Arguments */
  _trace_handle (SQL_HANDLE_DBC, ConnectionHandle);
  _trace_connattr_type (Attribute);
  _trace_pointer (ValuePtr);
  _trace_bufferlen (BufferLength);
  _trace_integer_p (StringLengthPtr, TRACE_OUTPUT_SUCCESS);
}
#endif
