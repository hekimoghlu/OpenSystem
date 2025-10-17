/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 13, 2023.
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
#ifndef _SCNETWORKSIGNATURE_H
#define _SCNETWORKSIGNATURE_H

#include <os/availability.h>
#include <sys/cdefs.h>
#include <CoreFoundation/CFString.h>
#include <CoreFoundation/CFArray.h>
#include <netinet/in.h>

__BEGIN_DECLS

/*!
	@header SCNetworkSignature
	@discussion The SCNetworkSignature API provides access to the
		network identification information.  Each routable network
		has an associated signature that is assigned a unique
		identifier.  The unique identifier allows an application
		to associate settings with a particular network
		or set of networks.
 */

/*!
	@function SCNetworkSignatureCopyActiveIdentifiers
	@discussion Find all currently active networks and return a list of
		(string) identifiers, one for each network.
	@param alloc The CFAllocator that should be used to allocate
		memory for the local dynamic store object.
		This parameter may be NULL in which case the current
		default CFAllocator is used. If this reference is not
		a valid CFAllocator, the behavior is undefined.
	@result A CFArrayRef containing a list of (string) identifiers,
		NULL if no networks are currently active.
 */
CFArrayRef /* of CFStringRef's */
SCNetworkSignatureCopyActiveIdentifiers(CFAllocatorRef alloc)			API_AVAILABLE(macos(10.5)) SPI_AVAILABLE(ios(2.0), tvos(9.0), watchos(1.0), bridgeos(1.0));

/*!
	@function SCNetworkSignatureCopyActiveIdentifierForAddress
	@discussion Find the one active network associated with the specified
		address and return the unique (string) identifier that
		represents it.
	@param alloc The CFAllocator that should be used to allocate
		memory for the local dynamic store object.
		This parameter may be NULL in which case the current
		default CFAllocator is used. If this reference is not
		a valid CFAllocator, the behavior is undefined.
	@param addr The address of interest.  Note: currently only AF_INET
		0.0.0.0 is supported, passing anything else always returns
		NULL.
	@result The unique (string) identifier associated with the address,
		NULL if no network is known.
 */
CFStringRef
SCNetworkSignatureCopyActiveIdentifierForAddress(CFAllocatorRef alloc,
						 const struct sockaddr * addr)	API_AVAILABLE(macos(10.5)) SPI_AVAILABLE(ios(2.0), tvos(9.0), watchos(1.0), bridgeos(1.0));

/*!
	@function SCNetworkSignatureCopyIdentifierForConnectedSocket
	@discussion Find the identifier for the given file descriptor
		corresponding to a connected socket.
	@param alloc The CFAllocator that should be used to allocate
		memory for the local dynamic store object.
		This parameter may be NULL in which case the current
		default CFAllocator is used. If this reference is not
		a valid CFAllocator, the behavior is undefined.
	@param sock_fd The socket file descriptor, must be either AF_INET
		or AF_INET6.
	@result The unique (string) identifier associated with the address,
		NULL if no network is known.
 */
CFStringRef
SCNetworkSignatureCopyIdentifierForConnectedSocket(CFAllocatorRef alloc,
						   int sock_fd) API_AVAILABLE(macos(10.7)) SPI_AVAILABLE(ios(5.0), tvos(9.0), watchos(1.0), bridgeos(1.0));

__END_DECLS

#endif	/* _SCNETWORKSIGNATURE_H */
