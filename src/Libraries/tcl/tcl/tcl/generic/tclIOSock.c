/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 20, 2022.
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
#include "tclInt.h"

/*
 *---------------------------------------------------------------------------
 *
 * TclSockGetPort --
 *
 *	Maps from a string, which could be a service name, to a port. Used by
 *	socket creation code to get port numbers and resolve registered
 *	service names to port numbers.
 *
 * Results:
 *	A standard Tcl result. On success, the port number is returned in
 *	portPtr. On failure, an error message is left in the interp's result.
 *
 * Side effects:
 *	None.
 *
 *---------------------------------------------------------------------------
 */

int
TclSockGetPort(
    Tcl_Interp *interp,
    const char *string, /* Integer or service name */
    const char *proto, /* "tcp" or "udp", typically */
    int *portPtr)		/* Return port number */
{
    struct servent *sp;		/* Protocol info for named services */
    Tcl_DString ds;
    const char *native;

    if (Tcl_GetInt(NULL, string, portPtr) != TCL_OK) {
	/*
	 * Don't bother translating 'proto' to native.
	 */

	native = Tcl_UtfToExternalDString(NULL, string, -1, &ds);
	sp = getservbyname(native, proto);		/* INTL: Native. */
	Tcl_DStringFree(&ds);
	if (sp != NULL) {
	    *portPtr = ntohs((unsigned short) sp->s_port);
	    return TCL_OK;
	}
    }
    if (Tcl_GetInt(interp, string, portPtr) != TCL_OK) {
	return TCL_ERROR;
    }
    if (*portPtr > 0xFFFF) {
	Tcl_AppendResult(interp, "couldn't open socket: port number too high",
		NULL);
	return TCL_ERROR;
    }
    return TCL_OK;
}

/*
 *----------------------------------------------------------------------
 *
 * TclSockMinimumBuffers --
 *
 *	Ensure minimum buffer sizes (non zero).
 *
 * Results:
 *	A standard Tcl result.
 *
 * Side effects:
 *	Sets SO_SNDBUF and SO_RCVBUF sizes.
 *
 *----------------------------------------------------------------------
 */

int
TclSockMinimumBuffers(
    int sock,			/* Socket file descriptor */
    int size)			/* Minimum buffer size */
{
    int current;
    socklen_t len;

    len = sizeof(int);
    getsockopt(sock, SOL_SOCKET, SO_SNDBUF, (char *)&current, &len);
    if (current < size) {
	len = sizeof(int);
	setsockopt(sock, SOL_SOCKET, SO_SNDBUF, (char *)&size, len);
    }
    len = sizeof(int);
    getsockopt(sock, SOL_SOCKET, SO_RCVBUF, (char *)&current, &len);
    if (current < size) {
	len = sizeof(int);
	setsockopt(sock, SOL_SOCKET, SO_RCVBUF, (char *)&size, len);
    }
    return TCL_OK;
}

/*
 * Local Variables:
 * mode: c
 * c-basic-offset: 4
 * fill-column: 78
 * End:
 */
