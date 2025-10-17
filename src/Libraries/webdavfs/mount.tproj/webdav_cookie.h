/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 23, 2023.
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
#ifndef webdavfs_webdav_cookie_h
#define webdavfs_webdav_cookie_h

#include <CoreFoundation/CoreFoundation.h>
#include <CoreServices/CoreServices.h>
#include "webdav.h"

typedef struct cookie_type
{
	// Cookie header for sending, in form "name=val"
	CFMutableStringRef cookie_header;
	
    CFStringRef cookie_name;
    char        *cookie_name_str;
    
    CFStringRef cookie_val;
    char        *cookie_val_str;
    
    CFStringRef cookie_path;
    char        *cookie_path_str;
    
    CFStringRef cookie_domain;
    char        *cookie_domain_str;
    
    boolean_t   has_expire_time;
    time_t      cookie_expire_time;
    
    boolean_t   cookie_secure;
    boolean_t   cookie_httponly;
	
	struct cookie_type *next, *prev;
} WEBDAV_COOKIE;

void cookies_init(void);
void add_cookie_headers(CFHTTPMessageRef message, CFURLRef url);
void handle_cookies(CFStringRef str, CFHTTPMessageRef message);
void purge_expired_cookies(void);

void dump_cookies(struct webdav_request_cookies *req);
void reset_cookies(struct webdav_request_cookies *req);


#endif
