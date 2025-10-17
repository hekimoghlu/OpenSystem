/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 10, 2024.
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
#ifndef _SCDTEST_H
#define _SCDTEST_H

/*
 * SCDynamicStore Test keys/entitlements
 */
#define SCDTEST_PREFIX			"com.apple.SCDynamicStore.test."
#define SCDTEST_PREFIX_PATTERN_STR	"com\\.apple\\.SCDynamicStore\\.test\\."
#define SCDTEST_ENTITLEMENT(a)		CFSTR(a ".entitlement")
#define SCDTEST_KEY(a)			CFSTR(a ".key")
#define SCDTEST_PREFIX_PATTERN(a)	CFSTR(SCDTEST_PREFIX_PATTERN_STR a)

/*
 * readDeny
 *   SCDTEST_READ_DENY{1,2}_KEY are not readable if process holds
 *   corresponding SCDTEST_READ_DENY{1,2}_ENTITLEMENT
 */
#define SCDTEST_READ_DENY		SCDTEST_PREFIX "read-deny"
#define SCDTEST_READ_DENY1		SCDTEST_READ_DENY "1"
#define SCDTEST_READ_DENY1_ENTITLEMENT 	SCDTEST_ENTITLEMENT(SCDTEST_READ_DENY1)
#define SCDTEST_READ_DENY1_KEY		SCDTEST_KEY(SCDTEST_READ_DENY1)

#define SCDTEST_READ_DENY2		SCDTEST_READ_DENY "2"
#define SCDTEST_READ_DENY2_ENTITLEMENT 	SCDTEST_ENTITLEMENT(SCDTEST_READ_DENY2)
#define SCDTEST_READ_DENY2_KEY		SCDTEST_KEY(SCDTEST_READ_DENY2)

/*
 * readAllow
 *   SCDTEST_READ_ALLOW{1,2}_KEY are only readable if process holds
 *   corresponding SCDTEST_READ_ALLOW{1,2}_ENTITLEMENT
 */
#define SCDTEST_READ_ALLOW		SCDTEST_PREFIX "read-allow"
#define SCDTEST_READ_ALLOW1		SCDTEST_READ_ALLOW "1"
#define SCDTEST_READ_ALLOW1_ENTITLEMENT	SCDTEST_ENTITLEMENT(SCDTEST_READ_ALLOW1)
#define SCDTEST_READ_ALLOW1_KEY		SCDTEST_KEY(SCDTEST_READ_ALLOW1)

#define SCDTEST_READ_ALLOW2		SCDTEST_READ_ALLOW "2"
#define SCDTEST_READ_ALLOW2_ENTITLEMENT	SCDTEST_ENTITLEMENT(SCDTEST_READ_ALLOW2)
#define SCDTEST_READ_ALLOW2_KEY		SCDTEST_KEY(SCDTEST_READ_ALLOW2)

/*
 * Read pattern
 */
#define SCDTEST_READ_PATTERN	SCDTEST_PREFIX_PATTERN("read.*")

/*
 * writeProtect
 *   SCDTEST_WRITE_PROTECT{1,2}_KEY are only writable if process holds the
 *   key-specific entitlement.
 */
#define SCDTEST_WRITE_PROTECT		SCDTEST_PREFIX "write-protect"
#define SCDTEST_WRITE_PROTECT1		SCDTEST_WRITE_PROTECT "1"
#define SCDTEST_WRITE_PROTECT1_KEY	SCDTEST_KEY(SCDTEST_WRITE_PROTECT1)

#define SCDTEST_WRITE_PROTECT2		SCDTEST_WRITE_PROTECT "2"
#define SCDTEST_WRITE_PROTECT2_KEY	SCDTEST_KEY(SCDTEST_WRITE_PROTECT2)
	
#endif /* _SCDTEST_H */
