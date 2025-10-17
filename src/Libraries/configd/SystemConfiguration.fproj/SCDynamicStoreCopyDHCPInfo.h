/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 20, 2021.
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
#ifndef _SCDYNAMICSTORECOPYDHCPINFO_H
#define _SCDYNAMICSTORECOPYDHCPINFO_H

#include <os/availability.h>
#include <sys/cdefs.h>
#include <CoreFoundation/CoreFoundation.h>
#include <SystemConfiguration/SCDynamicStore.h>

CF_IMPLICIT_BRIDGING_ENABLED
CF_ASSUME_NONNULL_BEGIN

/*!
	@header SCDynamicStoreCopyDHCPInfo
	@discussion The functions of the SCDynamicStoreCopyDHCPInfo API
		provide access to information returned by the DHCP or
		BOOTP server.
 */


__BEGIN_DECLS

/*!
	@function SCDynamicStoreCopyDHCPInfo
	@discussion Copies the DHCP information for the requested serviceID,
		or the primary service if serviceID == NULL.
	@param store An SCDynamicStoreRef representing the dynamic store session
		that should be used for communication with the server.
		If NULL, a temporary session will be used.
	@param serviceID A CFStringRef containing the requested service.
		If NULL, returns information for the primary service.
	@result Returns a dictionary containing DHCP information if successful;
		NULL otherwise.
		Use the DHCPInfoGetOption function to retrieve
		individual options from the returned dictionary.

		A non-NULL return value must be released using CFRelease().
 */
CFDictionaryRef __nullable
SCDynamicStoreCopyDHCPInfo	(SCDynamicStoreRef	__nullable	store,
				 CFStringRef		__nullable	serviceID)	API_AVAILABLE(macos(10.1)) SPI_AVAILABLE(ios(2.0), tvos(9.0), watchos(1.0), bridgeos(1.0));

/*!
	@function DHCPInfoGetOptionData
	@discussion Returns a non-NULL CFDataRef containing the DHCP
		option data, if present.
	@param info The non-NULL DHCP information dictionary returned by
		calling SCDynamicStoreCopyDHCPInfo.
	@param code The DHCP option code (see RFC 2132) to return
		data for.
	@result Returns a non-NULL CFDataRef containing the option data;
		NULL if the requested option data is not present.

		The return value must NOT be released.
 */
CFDataRef __nullable
DHCPInfoGetOptionData		(CFDictionaryRef	info,
				 UInt8			code)		API_AVAILABLE(macos(10.1)) SPI_AVAILABLE(ios(2.0), tvos(9.0), watchos(1.0), bridgeos(1.0));

/*!
	@function DHCPInfoGetLeaseStartTime
	@discussion Returns a CFDateRef corresponding to the lease start time,
		if present.
	@param info The non-NULL DHCP information dictionary returned by
		calling SCDynamicStoreCopyDHCPInfo.
	@result Returns a non-NULL CFDateRef if lease start time information is
		present; NULL if the information is not present or if the
		configuration method is not DHCP.

		The return value must NOT be released.
 */
CFDateRef __nullable
DHCPInfoGetLeaseStartTime	(CFDictionaryRef	info)		API_AVAILABLE(macos(10.1)) SPI_AVAILABLE(ios(2.0), tvos(9.0), watchos(1.0), bridgeos(1.0));


/*!
	@function DHCPInfoGetLeaseExpirationTime
	@discussion Returns a CFDateRef corresponding to the lease expiration time,
		if present.
	@param info The non-NULL DHCP information dictionary returned by
		calling SCDynamicStoreCopyDHCPInfo.
	@result Returns a non-NULL CFDateRef if the DHCP lease has an expiration;
		NULL if the lease is infinite i.e. has no expiration, or the
		configuration method is not DHCP. An infinite lease can be determined
		by a non-NULL lease start time (see DHCPInfoGetLeaseStartTime above).

		The return value must NOT be released.
*/
CFDateRef __nullable
DHCPInfoGetLeaseExpirationTime	(CFDictionaryRef	info)		API_AVAILABLE(macos(10.8)) SPI_AVAILABLE(ios(6.0), tvos(9.0), watchos(1.0), bridgeos(1.0));

__END_DECLS

CF_ASSUME_NONNULL_END
CF_IMPLICIT_BRIDGING_DISABLED

#endif	/* _SCDYNAMICSTORECOPYDHCPINFO_H */
