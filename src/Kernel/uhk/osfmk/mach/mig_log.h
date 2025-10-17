/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 13, 2023.
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
/*
 * @OSF_COPYRIGHT@
 */

#ifndef _mig_log_
#define _mig_log_

#include <sys/appleapiopts.h>

#ifdef __APPLE_API_OBSOLETE

typedef enum {
	MACH_MSG_LOG_USER,
	MACH_MSG_LOG_SERVER
} mig_who_t;

typedef enum {
	MACH_MSG_REQUEST_BEING_SENT,
	MACH_MSG_REQUEST_BEING_RCVD,
	MACH_MSG_REPLY_BEING_SENT,
	MACH_MSG_REPLY_BEING_RCVD
} mig_which_event_t;

typedef enum {
	MACH_MSG_ERROR_WHILE_PARSING,
	MACH_MSG_ERROR_UNKNOWN_ID
} mig_which_error_t;

extern void MigEventTracer
#if     defined(__STDC__)
(
	mig_who_t who,
	mig_which_event_t what,
	mach_msg_id_t msgh_id,
	unsigned int size,
	unsigned int kpd,
	unsigned int retcode,
	unsigned int ports,
	unsigned int oolports,
	unsigned int ool,
	char *file,
	unsigned int line
);
#else   /* !defined(__STDC__) */
();
#endif  /* !defined(__STDC__) */

extern void MigEventErrors
#if     defined(__STDC__)
(
	mig_who_t who,
	mig_which_error_t what,
	void *par,
	char *file,
	unsigned int line
);
#else   /* !defined(__STDC__) */
();
#endif  /* !defined(__STDC__) */

extern int mig_errors;
extern int mig_tracing;

#define LOG_ERRORS      if (mig_errors)  MigEventErrors
#define LOG_TRACE       if (mig_tracing) MigEventTracer

#endif  /* __APPLE_API_OBSOLETE */

#endif  /* _mach_log_ */
