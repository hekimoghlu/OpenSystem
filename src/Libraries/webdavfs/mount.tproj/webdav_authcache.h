/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 25, 2021.
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
#ifndef _WEBDAV_AUTHCACHE_H_INCLUDE
#define _WEBDAV_AUTHCACHE_H_INCLUDE

#include <sys/types.h>
#include <sys/queue.h>
#include <CoreServices/CoreServices.h>

/*****************************************************************************/

/*
 * Paths for the localization bundle and the generic server icon 
 */
#define WEBDAV_LOCALIZATION_BUNDLE "/System/Library/Filesystems/webdav.fs"

/*
 * The amount of time to leave an authorization dialog up before auto-dismissing it
 */
#define WEBDAV_AUTHENTICATION_TIMEOUT 300.0

/*****************************************************************************/

/*
 * Structure used to store context
 * while the mount is being authenticated.
 *
 */
struct authcache_request_ctx {
	UInt32 count;		/* tracks retries for a request */
	UInt32 generation;	/* the generation count of the cached entry */
};

typedef struct authcache_request_ctx AuthRequestContext;

int authcache_apply(
	uid_t uid,							/* -> uid of the user making the request */
	CFHTTPMessageRef request,			/* -> the request to apply authentication to */
	UInt32 statusCode,					/* -> the status code (401, 407), or 0 if no challenge */
	CFHTTPMessageRef response,			/* -> the response containing the challenge, or NULL if no challenge */
	UInt32 *generation);				/* <- the generation count of the cache entry */

int authcache_valid(
	uid_t uid,							/* -> uid of the user making the request */
	CFHTTPMessageRef request,			/* -> the message of the successful request */
	UInt32 generation);					/* -> the generation count of the cache entry */

int authcache_proxy_invalidate(void);

int authcache_init(
	char *username,				/* -> username to attempt to use on first server challenge, or NULL */
	char *password,				/* -> password to attempt to use on first server challenge, or NULL */
	char *proxy_username,		/* -> username to attempt to use on first proxy server challenge, or NULL */
	char *proxy_password,		/* -> password to attempt to use on first proxy server challenge, or NULL */
	char *domain);				/* -> account domain to attempt to use on first server challenge, or NULL */

/*****************************************************************************/

#endif
