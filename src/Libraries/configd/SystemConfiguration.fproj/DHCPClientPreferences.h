/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 8, 2021.
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
#ifndef _DHCPCLIENTPREFERENCES_H
#define _DHCPCLIENTPREFERENCES_H

#include <os/availability.h>
#include <sys/cdefs.h>
#include <CoreFoundation/CFString.h>

CF_IMPLICIT_BRIDGING_ENABLED
CF_ASSUME_NONNULL_BEGIN

/*!
	@header DHCPClientPreferences
	@discussion The DHCPClientPreferences API allows applications to get and update DHCP preferences.
		DHCP preferences are in the form of DHCP option codes, which are defined in RFC 2132.
 */

__BEGIN_DECLS

/*!
	@function DHCPClientPreferencesSetApplicationOptions
	@discussion Updates the DHCP client preferences to include the
		given list of options for the given application ID.
	@param applicationID The application's preference ID, for example:
		"com.apple.SystemPreferences".
	@param options An array of 8-bit values containing the
		DHCP option codes (see RFC 2132) for this application ID.
		A NULL value will clear the list of options for this
		application ID.
	@param count The number of elements in the options parameter.
	@result Returns TRUE if the operation succeeded, FALSE otherwise.
 */

Boolean
DHCPClientPreferencesSetApplicationOptions	(CFStringRef			applicationID,
						 const UInt8	* __nullable	options,
						 CFIndex			count)		API_AVAILABLE(macos(10.1)) SPI_AVAILABLE(ios(2.0), tvos(9.0), watchos(1.0), bridgeos(1.0));

/*!
	@function DHCPClientPreferencesCopyApplicationOptions
	@discussion Copies the requested DHCP options for the
		given application ID.
	@param applicationID The application's preference ID, for example
		"com.apple.SystemPreferences".
	@param count The number of elements in the returned array.
	@result Returns the list of options for the given application ID, or
		NULL if no options are defined or an error occurred.

		When you are finished, use free() to release a non-NULL return value.
 */

UInt8 * __nullable
DHCPClientPreferencesCopyApplicationOptions	(CFStringRef	applicationID,
						 CFIndex	*count)		API_AVAILABLE(macos(10.1)) SPI_AVAILABLE(ios(2.0), tvos(9.0), watchos(1.0), bridgeos(1.0));

__END_DECLS

CF_ASSUME_NONNULL_END
CF_IMPLICIT_BRIDGING_DISABLED

#endif	/* _DHCPCLIENTPREFERENCES_H */
