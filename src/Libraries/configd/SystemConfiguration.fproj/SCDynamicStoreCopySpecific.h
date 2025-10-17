/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 2, 2024.
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
#ifndef _SCDYNAMICSTORECOPYSPECIFIC_H
#define _SCDYNAMICSTORECOPYSPECIFIC_H

#include <os/availability.h>
#include <sys/cdefs.h>
#include <sys/types.h>
#include <CoreFoundation/CoreFoundation.h>
#include <SystemConfiguration/SCDynamicStore.h>

CF_IMPLICIT_BRIDGING_ENABLED
CF_ASSUME_NONNULL_BEGIN

/*!
	@header SCDynamicStoreCopySpecific
	@discussion The functions of the SCDynamicStoreCopySpecific API
		allow an application to determine specific configuration
		information about the current system (for example, the
		computer or sharing name, the currently logged-in user, etc.).
 */


__BEGIN_DECLS

/*!
	@function SCDynamicStoreCopyComputerName
	@discussion Gets the current computer name.
	@param store An SCDynamicStoreRef representing the dynamic store
		session that should be used for communication with the server.
		If NULL, a temporary session will be used.
	@param nameEncoding A pointer to memory that, if non-NULL, will be
		filled with the encoding associated with the computer or
		host name.
	@result Returns the current computer name;
		NULL if the name has not been set or if an error was encountered.
		You must release the returned value.
 */
CFStringRef __nullable
SCDynamicStoreCopyComputerName		(
					SCDynamicStoreRef	__nullable	store,
					CFStringEncoding	* __nullable	nameEncoding
					)			API_AVAILABLE(macos(10.1)) SPI_AVAILABLE(ios(2.0), tvos(9.0), watchos(1.0), bridgeos(1.0));

/*!
	@function SCDynamicStoreCopyConsoleUser
	@discussion Gets the name, user ID, and group ID of the currently
		logged-in user.

		Note: this function only provides information about the
		      primary console.  It does not provide any details
		      about console sessions that have fast user switched
		      out or about other consoles.
	@param store An SCDynamicStoreRef representing the dynamic store
		session that should be used for communication with the server.
		If NULL, a temporary session will be used.
	@param uid A pointer to memory that will be filled with the user ID
		of the current console user. If NULL, this value will not
		be returned.
	@param gid A pointer to memory that will be filled with the group ID
		of the current console user. If NULL, this value will not be
		returned.
	@result Returns the user currently logged into the system;
		NULL if no user is logged in or if an error was encountered.
		You must release the returned value.
 */
CFStringRef __nullable
SCDynamicStoreCopyConsoleUser		(
					SCDynamicStoreRef	__nullable	store,
					uid_t			* __nullable	uid,
					gid_t			* __nullable	gid
					)			API_AVAILABLE(macos(10.1)) API_UNAVAILABLE(ios, tvos, watchos, bridgeos);

/*!
	@function SCDynamicStoreCopyLocalHostName
	@discussion Gets the current local host name.
	@param store An SCDynamicStoreRef representing the dynamic store
		session that should be used for communication with the server.
		If NULL, a temporary session will be used.
	@result Returns the current local host name;
		NULL if the name has not been set or if an error was encountered.
		You must release the returned value.
 */
CFStringRef __nullable
SCDynamicStoreCopyLocalHostName		(
					SCDynamicStoreRef	__nullable	store
					)			API_AVAILABLE(macos(10.1)) SPI_AVAILABLE(ios(2.0), tvos(9.0), watchos(1.0), bridgeos(1.0));

/*!
	@function SCDynamicStoreCopyLocation
	@discussion Gets the current location identifier.
	@param store An SCDynamicStoreRef representing the dynamic store
		session that should be used for communication with the server.
		If NULL, a temporary session will be used.
	@result Returns a string representing the current location identifier;
		NULL if no location identifier has been defined or if an error
		was encountered.
		You must release the returned value.
 */
CFStringRef __nullable
SCDynamicStoreCopyLocation		(
					SCDynamicStoreRef	__nullable	store
					)			API_AVAILABLE(macos(10.1)) API_UNAVAILABLE(ios, tvos, watchos, bridgeos);

/*!
	@function SCDynamicStoreCopyProxies
	@discussion Gets the current internet proxy settings.
		The returned proxy settings dictionary includes:

		<TABLE BORDER>
		<TR>
			<TH>key</TD>
			<TH>type</TD>
			<TH>description</TD>
		</TR>
		<TR>
			<TD>kSCPropNetProxiesExceptionsList</TD>
			<TD>CFArray[CFString]</TD>
			<TD>Host name patterns which should bypass the proxy</TD>
		</TR>
		<TR>
			<TD>kSCPropNetProxiesHTTPEnable</TD>
			<TD>CFNumber (0 or 1)</TD>
			<TD>Enables/disables the use of an HTTP proxy</TD>
		</TR>
		<TR>
			<TD>kSCPropNetProxiesHTTPProxy</TD>
			<TD>CFString</TD>
			<TD>The proxy host</TD>
		</TR>
		<TR>
			<TD>kSCPropNetProxiesHTTPPort</TD>
			<TD>CFNumber</TD>
			<TD>The proxy port number</TD>
		</TR>
		<TR>
			<TD>kSCPropNetProxiesHTTPSEnable</TD>
			<TD>CFNumber (0 or 1)</TD>
			<TD>Enables/disables the use of an HTTPS proxy</TD>
		</TR>
		<TR>
			<TD>kSCPropNetProxiesHTTPSProxy</TD>
			<TD>CFString</TD>
			<TD>The proxy host</TD>
		</TR>
		<TR>
			<TD>kSCPropNetProxiesHTTPSPort</TD>
			<TD>CFNumber</TD>
			<TD>The proxy port number</TD>
		</TR>
		<TR>
			<TD>kSCPropNetProxiesFTPEnable</TD>
			<TD>CFNumber (0 or 1)</TD>
			<TD>Enables/disables the use of an FTP proxy</TD>
		</TR>
		<TR>
			<TD>kSCPropNetProxiesFTPProxy</TD>
			<TD>CFString</TD>
			<TD>The proxy host</TD>
		</TR>
		<TR>
			<TD>kSCPropNetProxiesFTPPort</TD>
			<TD>CFNumber</TD>
			<TD>The proxy port number</TD>
		</TR>
		<TR>
			<TD>kSCPropNetProxiesFTPPassive</TD>
			<TD>CFNumber (0 or 1)</TD>
			<TD>Enable passive mode operation for use behind connection
			filter-ing firewalls.</TD>
		</TR>
		</TABLE>

		Other key-value pairs are defined in the SCSchemaDefinitions.h
		header file.
	@param store An SCDynamicStoreRef representing the dynamic store
		session that should be used for communication with the server.
		If NULL, a temporary session will be used.
	@result Returns a dictionary containing key-value pairs that represent
		the current internet proxy settings;
		NULL if no proxy settings have been defined or if an error
		was encountered.
		You must release the returned value.
 */
CFDictionaryRef __nullable
SCDynamicStoreCopyProxies		(
					SCDynamicStoreRef	__nullable	store
					)			API_AVAILABLE(macos(10.1)) SPI_AVAILABLE(ios(2.0), tvos(9.0), watchos(1.0), bridgeos(1.0));

__END_DECLS

CF_ASSUME_NONNULL_END
CF_IMPLICIT_BRIDGING_DISABLED

#endif	/* _SCDYNAMICSTORECOPYSPECIFIC_H */
