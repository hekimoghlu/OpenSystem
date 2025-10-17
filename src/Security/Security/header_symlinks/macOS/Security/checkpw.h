/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 16, 2023.
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
#ifndef __CHKUSRNAMPASSWD_H__
#define __CHKUSRNAMPASSWD_H__

#include <Availability.h>

#include <pwd.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
	CHECKPW_SUCCESS = 0,
	CHECKPW_UNKNOWNUSER = -1,
	CHECKPW_BADPASSWORD = -2,
	CHECKPW_FAILURE = -3
};

/*!
	@function checkpw

	checks a username/password combination.

	@param userName (input) username as a UTF8 string
	@param password (input) password as a UTF8 string

	@result CHECKPW_SUCCESS username/password correct
	CHECKPW_UNKNOWNUSER no such user
	CHECKPW_BADPASSWORD wrong password
	CHECKPW_FAILURE failed to communicate with DirectoryServices

	@discussion Deprecated and should no longer be used.
	Username/password combinations can be checked in two ways:
	1) PAM(3): with the "checkpw" service.
	2) OpenDirectory: ODRecordVerifyPassword() - if you are
           currently using OpenDirectory.
*/

int checkpw( const char* userName, const char* password )
	__OSX_AVAILABLE_BUT_DEPRECATED(__MAC_10_1,__MAC_10_7,__IPHONE_NA,__IPHONE_NA);

#ifdef __cplusplus
}
#endif

#endif // __CHKUSRNAMPASSWD_H__
