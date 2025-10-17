/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 10, 2024.
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
#ifndef _SCPREFERENCESPATH_H
#define _SCPREFERENCESPATH_H

#include <os/availability.h>
#include <sys/cdefs.h>
#include <CoreFoundation/CoreFoundation.h>
#include <SystemConfiguration/SCPreferences.h>

CF_IMPLICIT_BRIDGING_ENABLED
CF_ASSUME_NONNULL_BEGIN

/*!
	@header SCPreferencesPath
	@discussion The SCPreferencesPath API allows an application to
		load and store XML configuration data in a controlled
		manner and provide the necessary notifications to other
		applications that need to be aware of configuration
		changes.

		The functions in the SCPreferencesPath API make certain
		assumptions about the layout of the preferences data.
		These functions view the data as a collection of dictionaries
		of key-value pairs and an associated path name.
		The root path ("/") identifies the top-level dictionary.
		Additional path components specify the keys for subdictionaries.

		For example, the following dictionary can be accessed via
		two paths.  The root ("/") path would return a dictionary
		with all keys and values.  The path "/path1" would only
		return the dictionary with the "key3" and "key4" properties.

	<pre>
	@textblock
	<dict>
		<key>key1</key>
		<string>val1</string>
		<key>key2</key>
		<string>val2</string>
		<key>path1</key>
		<dict>
			<key>key3</key>
			<string>val3</string>
			<key>key4</key>
			<string>val4</string>
		</dict>
	</dict>
	@/textblock
	</pre>

	Each dictionary can also include the kSCResvLink ("__LINK__") key.
	The value associated with this key is interpreted as a link to
	another path.  If this key is present, a call to the
	SCPreferencesPathGetValue function returns the dictionary
	specified by the link.
 */


__BEGIN_DECLS

/*!
	@function SCPreferencesPathCreateUniqueChild
	@discussion Creates a new path component within the dictionary
		hierarchy.
	@param prefs The preferences session.
	@param prefix A string that represents the parent path.
	@result Returns a string representing the new (unique) child path; NULL
		if the specified path does not exist.
 */
CFStringRef __nullable
SCPreferencesPathCreateUniqueChild	(
					SCPreferencesRef	prefs,
					CFStringRef		prefix
					)			API_AVAILABLE(macos(10.1)) SPI_AVAILABLE(ios(2.0), tvos(9.0), watchos(1.0), bridgeos(1.0));

/*!
	@function SCPreferencesPathGetValue
	@discussion Returns the dictionary associated with the specified
		path.
	@param prefs The preferences session.
	@param path A string that represents the path to be returned.
	@result Returns the dictionary associated with the specified path; NULL
		if the path does not exist.
 */
CFDictionaryRef __nullable
SCPreferencesPathGetValue		(
					SCPreferencesRef	prefs,
					CFStringRef		path
					)			API_AVAILABLE(macos(10.1)) SPI_AVAILABLE(ios(2.0), tvos(9.0), watchos(1.0), bridgeos(1.0));

/*!
	@function SCPreferencesPathGetLink
	@discussion Returns the link (if one exists) associated with the
		specified path.
	@param prefs The preferences session.
	@param path A string that represents the path to be returned.
	@result Returns the dictionary associated with the specified path; NULL
		if the path is not a link or does not exist.
 */
CFStringRef __nullable
SCPreferencesPathGetLink		(
					SCPreferencesRef	prefs,
					CFStringRef		path
					)			API_AVAILABLE(macos(10.1)) SPI_AVAILABLE(ios(2.0), tvos(9.0), watchos(1.0), bridgeos(1.0));

/*!
	@function SCPreferencesPathSetValue
	@discussion Associates a dictionary with the specified path.
	@param prefs The preferences session.
	@param path A string that represents the path to be updated.
	@param value A dictionary that represents the data to be
		stored at the specified path.
	@result Returns TRUE if successful; FALSE otherwise.
 */
Boolean
SCPreferencesPathSetValue		(
					SCPreferencesRef	prefs,
					CFStringRef		path,
					CFDictionaryRef		value
					)			API_AVAILABLE(macos(10.1)) SPI_AVAILABLE(ios(2.0), tvos(9.0), watchos(1.0), bridgeos(1.0));

/*!
	@function SCPreferencesPathSetLink
	@discussion Associates a link to a second dictionary at the
		specified path.
	@param prefs The preferences session.
	@param path A string that represents the path to be updated.
	@param link A string that represents the link to be stored
		at the specified path.
	@result Returns TRUE if successful; FALSE otherwise.
 */
Boolean
SCPreferencesPathSetLink		(
					SCPreferencesRef	prefs,
					CFStringRef		path,
					CFStringRef		link
					)			API_AVAILABLE(macos(10.1)) SPI_AVAILABLE(ios(2.0), tvos(9.0), watchos(1.0), bridgeos(1.0));

/*!
	@function SCPreferencesPathRemoveValue
	@discussion Removes the data associated with the specified path.
	@param prefs The preferences session.
	@param path A string that represents the path to be returned.
	@result Returns TRUE if successful; FALSE otherwise.
 */
Boolean
SCPreferencesPathRemoveValue		(
					SCPreferencesRef	prefs,
					CFStringRef		path
					)			API_AVAILABLE(macos(10.1)) SPI_AVAILABLE(ios(2.0), tvos(9.0), watchos(1.0), bridgeos(1.0));

__END_DECLS

CF_ASSUME_NONNULL_END
CF_IMPLICIT_BRIDGING_DISABLED

#endif	/* _SCPREFERENCESPATH_H */
