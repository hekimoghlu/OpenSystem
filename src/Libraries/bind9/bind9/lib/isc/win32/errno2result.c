/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 9, 2022.
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
/* $Id: errno2result.c,v 1.17 2008/09/12 04:46:25 marka Exp $ */

#include <config.h>

#include <winsock2.h>
#include "errno2result.h"
#include <isc/result.h>
#include <isc/strerror.h>
#include <isc/util.h>

/*
 * Convert a POSIX errno value into an isc_result_t.  The
 * list of supported errno values is not complete; new users
 * of this function should add any expected errors that are
 * not already there.
 */
isc_result_t
isc__errno2resultx(int posixerrno, isc_boolean_t dolog,
		   const char *file, int line)
{
	char strbuf[ISC_STRERRORSIZE];

	switch (posixerrno) {
	case ENOTDIR:
	case WSAELOOP:
	case WSAEINVAL:
	case EINVAL:		/* XXX sometimes this is not for files */
	case ENAMETOOLONG:
	case WSAENAMETOOLONG:
	case EBADF:
	case WSAEBADF:
		return (ISC_R_INVALIDFILE);
	case ENOENT:
		return (ISC_R_FILENOTFOUND);
	case EACCES:
	case WSAEACCES:
	case EPERM:
		return (ISC_R_NOPERM);
	case EEXIST:
		return (ISC_R_FILEEXISTS);
	case EIO:
		return (ISC_R_IOERROR);
	case ENOMEM:
		return (ISC_R_NOMEMORY);
#ifdef EOVERFLOW
	case EOVERFLOW:
		return (ISC_R_RANGE);
#endif
	case ENFILE:
	case EMFILE:
	case WSAEMFILE:
		return (ISC_R_TOOMANYOPENFILES);
	case ERROR_CANCELLED:
		return (ISC_R_CANCELED);
	case ERROR_CONNECTION_REFUSED:
	case WSAECONNREFUSED:
		return (ISC_R_CONNREFUSED);
	case WSAENOTCONN:
	case ERROR_CONNECTION_INVALID:
		return (ISC_R_NOTCONNECTED);
	case ERROR_HOST_UNREACHABLE:
	case WSAEHOSTUNREACH:
		return (ISC_R_HOSTUNREACH);
	case ERROR_NETWORK_UNREACHABLE:
	case WSAENETUNREACH:
		return (ISC_R_NETUNREACH);
	case ERROR_NO_NETWORK:
		return (ISC_R_NETUNREACH);
	case ERROR_PORT_UNREACHABLE:
		return (ISC_R_HOSTUNREACH);
	case ERROR_SEM_TIMEOUT:
		return (ISC_R_TIMEDOUT);
	case WSAECONNRESET:
	case WSAENETRESET:
	case WSAECONNABORTED:
	case WSAEDISCON:
	case ERROR_OPERATION_ABORTED:
	case ERROR_CONNECTION_ABORTED:
	case ERROR_REQUEST_ABORTED:
		return (ISC_R_CONNECTIONRESET);
	case WSAEADDRNOTAVAIL:
		return (ISC_R_ADDRNOTAVAIL);
	case ERROR_NETNAME_DELETED:
	case WSAENETDOWN:
		return (ISC_R_NETUNREACH);
	case WSAEHOSTDOWN:
		return (ISC_R_HOSTUNREACH);
	case WSAENOBUFS:
		return (ISC_R_NORESOURCES);
	default:
		if (dolog) {
			isc__strerror(posixerrno, strbuf, sizeof(strbuf));
			UNEXPECTED_ERROR(file, line,
					 "unable to convert errno "
					 "to isc_result: %d: %s",
					 posixerrno, strbuf);
		}
		/*
		 * XXXDCL would be nice if perhaps this function could
		 * return the system's error string, so the caller
		 * might have something more descriptive than "unexpected
		 * error" to log with.
		 */
		return (ISC_R_UNEXPECTED);
	}
}
