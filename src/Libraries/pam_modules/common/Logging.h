/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 6, 2025.
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

//
//  Logging.h
//  pam_modules
//

#ifndef Logging_h
#define Logging_h

// to turn on os_logs, uncomment following line
// #define PAM_USE_OS_LOG

#ifdef PAM_USE_OS_LOG
#include <os/log.h>
#include <os/activity.h>

#define PAM_DEFINE_LOG(category) \
static os_log_t PAM_LOG_ ## category () { \
static dispatch_once_t once; \
static os_log_t log; \
dispatch_once(&once, ^{ log = os_log_create("com.apple.pam", #category); }); \
return log; \
};

#define _LOG_DEBUG(...) os_log_debug(PAM_LOG, __VA_ARGS__)
#define _LOG_VERBOSE(...) os_log_debug(PAM_LOG, __VA_ARGS__)
#define _LOG_INFO(...) os_log_info(PAM_LOG, __VA_ARGS__)
#define _LOG_DEFAULT(...) os_log(PAM_LOG, __VA_ARGS__)
#define _LOG_ERROR(...) os_log_error(PAM_LOG, __VA_ARGS__)

#else

#define _LOG_DEBUG(...) openpam_log(PAM_LOG_DEBUG, __VA_ARGS__)
#define _LOG_VERBOSE(...) openpam_log(PAM_LOG_VERBOSE, __VA_ARGS__)
#define _LOG_INFO(...) openpam_log(PAM_LOG_VERBOSE, __VA_ARGS__)
#define _LOG_DEFAULT(...) openpam_log(PAM_LOG_NOTICE, __VA_ARGS__)
#define _LOG_ERROR(...) openpam_log(PAM_LOG_ERROR, __VA_ARGS__)
#endif

#endif /* Logging_h */
