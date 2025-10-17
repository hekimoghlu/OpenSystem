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
/*
 * EAPLog.c
 * - functions to log EAP-related information
 */
/* 
 * Modification History
 *
 * December 26, 2012	Dieter Siegmund (dieter@apple.com)
 * - created
 */

#include <dispatch/dispatch.h>
#include <SystemConfiguration/SCPrivate.h>
#include "symbol_scope.h"
#include "EAPLog.h"

STATIC os_log_t S_eap_logger = NULL;

#define EAPOL_OS_LOG_SUBSYSTEM	"com.apple.eapol"

STATIC char * const S_eap_os_log_categories[] = {
	"Controller",
	"Monitor",
	"Client",
	"Framework"
};

os_log_t
EAPLogGetLogHandle()
{
    if (S_eap_logger == NULL) {
	EAPLogInit(kEAPLogCategoryFramework);
    }
    return S_eap_logger;
}

#define CHECK_LOG_LEVEL_LIMIT(log_category)				\
	do {												\
	if (log_category < kEAPLogCategoryController ||		\
		log_category > kEAPLogCategoryFramework) {		\
		return;											\
	}													\
	} while(0)

void
EAPLogInit(EAPLogCategory log_category)
{
	CHECK_LOG_LEVEL_LIMIT(log_category);
	S_eap_logger = os_log_create(EAPOL_OS_LOG_SUBSYSTEM,
								 S_eap_os_log_categories[log_category]);
}
