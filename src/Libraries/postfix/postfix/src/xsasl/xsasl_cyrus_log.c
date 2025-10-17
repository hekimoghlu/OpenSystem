/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 10, 2022.
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
/* System library. */

#include <sys_defs.h>

/* Utility library. */

#include <msg.h>

/* Application-specific */

#include <xsasl_cyrus_common.h>

#if defined(USE_SASL_AUTH) && defined(USE_CYRUS_SASL)

#include <sasl.h>
#include <saslutil.h>

/* xsasl_cyrus_log - logging callback */

int     xsasl_cyrus_log(void *unused_context, int priority,
			        const char *message)
{
    switch (priority) {
	case SASL_LOG_ERR:		/* unusual errors */
#ifdef SASL_LOG_WARN			/* non-fatal warnings (Cyrus-SASL v2) */
	case SASL_LOG_WARN:
#endif
#ifdef SASL_LOG_WARNING			/* non-fatal warnings (Cyrus-SASL v1) */
	case SASL_LOG_WARNING:
#endif
	msg_warn("SASL authentication problem: %s", message);
	break;
#ifdef SASL_LOG_INFO
    case SASL_LOG_INFO:			/* other info (Cyrus-SASL v1) */
	if (msg_verbose)
	    msg_info("SASL authentication info: %s", message);
	break;
#endif
#ifdef SASL_LOG_NOTE
    case SASL_LOG_NOTE:			/* other info (Cyrus-SASL v2) */
	if (msg_verbose)
	    msg_info("SASL authentication info: %s", message);
	break;
#endif
#ifdef SASL_LOG_FAIL
    case SASL_LOG_FAIL:			/* authentication failures
						 * (Cyrus-SASL v2) */
	msg_warn("SASL authentication failure: %s", message);
	break;
#endif
#ifdef SASL_LOG_DEBUG
    case SASL_LOG_DEBUG:			/* more verbose than LOG_NOTE
						 * (Cyrus-SASL v2) */
	if (msg_verbose > 1)
	    msg_info("SASL authentication debug: %s", message);
	break;
#endif
#ifdef SASL_LOG_TRACE
    case SASL_LOG_TRACE:			/* traces of internal
						 * protocols (Cyrus-SASL v2) */
	if (msg_verbose > 1)
	    msg_info("SASL authentication trace: %s", message);
	break;
#endif
#ifdef SASL_LOG_PASS
    case SASL_LOG_PASS:			/* traces of internal
						 * protocols, including
						 * passwords (Cyrus-SASL v2) */
	if (msg_verbose > 1)
	    msg_info("SASL authentication pass: %s", message);
	break;
#endif
    }
    return (SASL_OK);
}

#endif
