/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 26, 2024.
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
#ifndef _SECURITY_AUTH_CCAUDIT_H_
#define _SECURITY_AUTH_CCAUDIT_H_

#include <bsm/audit_uevents.h>

AUTH_WARN_RESULT AUTH_MALLOC AUTH_NONNULL_ALL AUTH_RETURNS_RETAINED
ccaudit_t ccaudit_create(process_t, auth_token_t, int32_t event);

AUTH_NONNULL_ALL
void ccaudit_log_authorization(ccaudit_t, const char * right, OSStatus err);

AUTH_NONNULL_ALL
void ccaudit_log_success(ccaudit_t, credential_t cred, const char * right);

AUTH_NONNULL_ALL
void ccaudit_log_failure(ccaudit_t, const char * credName, const char * right);

AUTH_NONNULL1
void ccaudit_log_mechanism(ccaudit_t, const char * right, const char * mech, uint32_t status, const char * interrupted);

AUTH_NONNULL1
void ccaudit_log(ccaudit_t, const char * right, const char * msg, OSStatus err);

#endif /* !_SECURITY_AUTH_CCAUDIT_H_ */
