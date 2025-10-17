/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 27, 2022.
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
/*
 *  AuthorizationTags.h -- Right tags for implementing access control in
 *  applications and daemons
 */

#ifndef _SECURITY_AUTHORIZATIONTAGS_H_
#define _SECURITY_AUTHORIZATIONTAGS_H_


/*!
	@header AuthorizationTags

	This header defines some of the supported rights tags to be used in the Authorization API.
*/


/*!
	@define kAuthorizationEnvironmentUsername
	The name of the AuthorizationItem that should be passed into the environment when specifying a username.  The value and valueLength should contain the username itself.
*/
#define kAuthorizationEnvironmentUsername  "username"

/*!
	@define kAuthorizationEnvironmentPassword
	The name of the AuthorizationItem that should be passed into the environment when specifying a password for a given username.  The value and valueLength should contain the actual password data.
*/
#define kAuthorizationEnvironmentPassword  "password"

/*!
	@define kAuthorizationEnvironmentShared
	The name of the AuthorizationItem that should be passed into the environment when specifying a username and password.  Adding this entry to the environment will cause the username/password to be added to the shared credential pool of the calling applications session.  This means that further calls by other applications in this session will automatically have this credential availible to them.  The value is ignored.
*/
#define kAuthorizationEnvironmentShared  "shared"

/*!
	@define kAuthorizationRightExecute
	The name of the AuthorizationItem that should be passed into the rights when preauthorizing for a call to AuthorizationExecuteWithPrivileges().
	
	You need to acquire this right to be able to perform a AuthorizationExecuteWithPrivileges() operation.  In addtion to this right you should obtain whatever rights the tool you are executing with privileges need to perform it's operation on your behalf.  Currently no options are supported but you should pass in the full path of the tool you wish to execute in the value and valueLength fields.  In the future we will limit the right to only execute the requested path, and we will display this information to the user.
*/
#define kAuthorizationRightExecute "system.privilege.admin"

/*!
	@define kAuthorizationEnvironmentPrompt
	The name of the AuthorizationItem that should be passed into the environment when specifying a invocation specific additional text.  The value should be a localized UTF8 string.
*/
#define kAuthorizationEnvironmentPrompt  "prompt"

/*!
	@define kAuthorizationEnvironmentIcon
	The name of the AuthorizationItem that should be passed into the environment when specifying an alternate icon to be used.  The value should be a full path to and image NSImage can deal with.
*/
#define kAuthorizationEnvironmentIcon  "icon"

/*!
    @define kAuthorizationPamResult
    Return code provided by PAM module
 */
#define kAuthorizationPamResult  "pam_result"

/*!
    @define kAuthorizationFlags
    Flags passed to AuthorizationCopyRights
 */
#define kAuthorizationFlags  "flags"



#endif /* !_SECURITY_AUTHORIZATIONTAGS_H_ */
