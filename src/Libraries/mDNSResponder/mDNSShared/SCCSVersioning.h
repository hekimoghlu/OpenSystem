/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 22, 2024.
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
#ifndef SCCVersioning_H
#define SCCVersioning_H

#include "general.h"

/*!
 *	@brief
 *		Evaluates to a string literal based on mDNSResponderVersion for the build info part of an SCCS-style
 *		version string.
 *
 *	@discussion
 *		This string is appended to MDNSRESPONDER_SCCS_VERSION_BASE().
 */
#ifdef mDNSResponderVersion
	#define MDNSRESPONDER_SCCS_VERSION_PART_BUILD_INFO()	" " MDNS_STRINGIFY(mDNSResponderVersion)
#else
	#define MDNSRESPONDER_SCCS_VERSION_PART_BUILD_INFO()	""
#endif

/*!
 *	@brief
 *		Evaluates to a string literal based on mDNSResponderVersion for the build info part of an SCCS-style
 *		version string.
 *
 *	@discussion
 *		Similar to MDNSRESPONDER_SCCS_VERSION_PART_BUILD_INFO() except that if mDNSResponderVersion is defined,
 *		then the stringified version of mDNSResponderVersion is prefixed by a hyphen.
 *
 *		If mDNSResponderVersion is not defined, then an unhyphenated " (Engineering Build)" is used instead.
 */
#ifdef mDNSResponderVersion
	#define MDNSRESPONDER_SCCS_VERSION_PART_BUILD_INFO_HYPHENATED()	"-" MDNS_STRINGIFY(mDNSResponderVersion)
#else
	#define MDNSRESPONDER_SCCS_VERSION_PART_BUILD_INFO_HYPHENATED()	" (Engineering Build)"
#endif

/*!
 *	@brief
 *		Evaluates to a string literal for the build time part of an SCCS-style version string.
 *
 *	@discussion
 *		If MDNS_VERSIONSTR_NODTS is defined and evaluates to non-zero, then an empty string is used instead. See
 *		rdar://5458929 for background on MDNS_VERSIONSTR_NODTS.
 */
#if defined(MDNS_VERSIONSTR_NODTS) && MDNS_VERSIONSTR_NODTS
	#define MDNSRESPONDER_SCCS_VERSION_PART_BUILD_TIME()	""
#else
	#define MDNSRESPONDER_SCCS_VERSION_PART_BUILD_TIME()	" (" __DATE__ " " __TIME__ ")"
#endif

/*!
 *	@brief
 *		Evaluates to a string literal that can be used as an SCCS-style version string.
 *
 *	@param PROGRAM_NAME
 *		The name of the program, which is the first part of the string.
 *
 *	@discussion
 *		The "@(#)" prefix is the special sequence that the `what` utility uses to indentify the version string.
 */
#define MDNSRESPONDER_SCCS_VERSION_BASE(PROGRAM_NAME)	"@(#) " #PROGRAM_NAME

/*!
 *	@brief
 *		Evaluates to a string literal that can be used as an SCCS-style version string.
 *
 *	@param PROGRAM_NAME
 *		The name of the program, which is the first part of the string.
 *
 *	@discussion
 *		The format of the string is "<program name> <build info> <build time>".
 */
#define MDNSRESPONDER_SCCS_VERSION_STRING(PROGRAM_NAME)	\
	MDNSRESPONDER_SCCS_VERSION_BASE(PROGRAM_NAME)		\
	MDNSRESPONDER_SCCS_VERSION_PART_BUILD_INFO()		\
	MDNSRESPONDER_SCCS_VERSION_PART_BUILD_TIME()

/*!
 *	@brief
 *		Evaluates to a string literal that can be used as an SCCS-style version string.
 *
 *	@param PROGRAM_NAME
 *		The name of the program, which is the first part of the string.
 *
 *	@discussion
 *		The format of the string is "<program name>-<build info> (<build time>)".
 */
#define MDNSRESPONDER_SCCS_VERSION_STRING_HYPHENATED(PROGRAM_NAME)	\
	MDNSRESPONDER_SCCS_VERSION_BASE(PROGRAM_NAME)					\
	MDNSRESPONDER_SCCS_VERSION_PART_BUILD_INFO_HYPHENATED()			\
	MDNSRESPONDER_SCCS_VERSION_PART_BUILD_TIME()

#endif	// SCCVersioning_H
