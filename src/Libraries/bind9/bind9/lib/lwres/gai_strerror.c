/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 21, 2022.
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
/* $Id: gai_strerror.c,v 1.22 2007/06/19 23:47:22 tbox Exp $ */

/*! \file gai_strerror.c
 * lwres_gai_strerror() returns an error message corresponding to an
 * error code returned by getaddrinfo(). The following error codes and
 * their meaning are defined in \link netdb.h include/lwres/netdb.h.\endlink
 *
 * \li #EAI_ADDRFAMILY address family for hostname not supported
 * \li #EAI_AGAIN temporary failure in name resolution
 * \li #EAI_BADFLAGS invalid value for #ai_flags
 * \li #EAI_FAIL non-recoverable failure in name resolution
 * \li #EAI_FAMILY ai_family not supported
 * \li #EAI_MEMORY memory allocation failure
 * \li #EAI_NODATA no address associated with hostname
 * \li #EAI_NONAME hostname or servname not provided, or not known
 * \li #EAI_SERVICE servname not supported for ai_socktype
 * \li #EAI_SOCKTYPE ai_socktype not supported
 * \li #EAI_SYSTEM system error returned in errno
 *
 * The message invalid error code is returned if ecode is out of range.
 *
 * ai_flags, ai_family and ai_socktype are elements of the struct
 * addrinfo used by lwres_getaddrinfo().
 *
 * \section gai_strerror_see See Also
 *
 * strerror, lwres_getaddrinfo(), getaddrinfo(), RFC2133.
 */

#include <config.h>

#include <lwres/netdb.h>

/*% Text of error messages. */
static const char *gai_messages[] = {
	"no error",
	"address family for hostname not supported",
	"temporary failure in name resolution",
	"invalid value for ai_flags",
	"non-recoverable failure in name resolution",
	"ai_family not supported",
	"memory allocation failure",
	"no address associated with hostname",
	"hostname nor servname provided, or not known",
	"servname not supported for ai_socktype",
	"ai_socktype not supported",
	"system error returned in errno",
	"bad hints",
	"bad protocol",
	"overflow"
};

/*% Returns an error message corresponding to an error code returned by getaddrinfo() */
char *
lwres_gai_strerror(int ecode) {
	union {
		const char *const_ptr;
		char *deconst_ptr;
	} ptr;

	if ((ecode < 0) ||
	    (ecode >= (int)(sizeof(gai_messages)/sizeof(*gai_messages))))
		ptr.const_ptr = "invalid error code";
	else
		ptr.const_ptr = gai_messages[ecode];
	return (ptr.deconst_ptr);
}
