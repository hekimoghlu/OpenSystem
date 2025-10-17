/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 24, 2022.
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
#ifndef __DESC_H
#define __DESC_H

#define APP_ROW_DESC	0
#define APP_PARAM_DESC	1
#define IMP_ROW_DESC	2
#define IMP_PARAM_DESC	3

typedef struct DESC_s {

  int type;
  HERR herr;   		/* list of descriptor errors */
  SQLRETURN rc;
  
  struct DESC_s *next;

  SQLHDBC hdbc;	 	/* connection associated with the descriptor */
  SQLHDESC dhdesc; 	/* the driver's desc handle */
  HSTMT hstmt;   	/* if not null - the descriptor is implicit to that statement */

  SWORD desc_cip;        /* Call in Progess flag */

  SQLSMALLINT err_rec;
} DESC_t;

#ifndef HDESC
#define HDESC SQLHDESC
#endif


#define IS_VALID_HDESC(x) \
	((x) != SQL_NULL_HDESC && \
	 ((DESC_t *)(x))->type == SQL_HANDLE_DESC && \
	 ((DESC_t *)(x))->hdbc != SQL_NULL_HDBC)


#define ENTER_DESC(hdesc, trace) \
	DESC (pdesc, hdesc); \
	SQLRETURN retcode = SQL_SUCCESS; \
        ODBC_LOCK();\
	TRACE(trace); \
    	if (!IS_VALID_HDESC (pdesc)) \
	  { \
	    retcode = SQL_INVALID_HANDLE; \
	    goto done; \
	  } \
	else if (pdesc->desc_cip) \
          { \
	    PUSHSQLERR (pdesc->herr, en_S1010); \
	    retcode = SQL_ERROR; \
	    goto done; \
	  } \
	pdesc->desc_cip = 1; \
	CLEAR_ERRORS (pdesc); \
	ODBC_UNLOCK()


#define LEAVE_DESC(hdesc, trace) \
	ODBC_LOCK (); \
    done: \
    	TRACE(trace); \
	pdesc->desc_cip = 0; \
	ODBC_UNLOCK (); \
	return (retcode)

#endif /* __DESC_H */
