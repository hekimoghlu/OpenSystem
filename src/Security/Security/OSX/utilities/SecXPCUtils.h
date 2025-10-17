/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 27, 2024.
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
/* C interfaces to Foundation XPC utilities */
#ifndef _UTILITIES_SECXPCUTILS_H_
#define _UTILITIES_SECXPCUTILS_H_

#include <CoreFoundation/CoreFoundation.h>
#include <xpc/xpc.h>

#ifdef __cplusplus
extern "C" {
#endif

CF_ASSUME_NONNULL_BEGIN

/* SecXPCClientCanEditPreferenceOwnership identifies whether the calling
   process has a keychain-access-groups entitlement of "*", which provides
   a hint to the caller that its role is one which edits item ownership.
   This hint is advisory only and is not used to grant access.
*/
Boolean SecXPCClientCanEditPreferenceOwnership(void);

/* SecXPCCopyClientApplicationIdentifier is designed only to return a
   string which identifies the calling application. In the case where
   there is no current XPC connection, the check is performed in-process.
   As such, this identifier is advisory only and should not be used as
   a security boundary.
*/
CFStringRef SecXPCCopyClientApplicationIdentifier(void);

CF_ASSUME_NONNULL_END

#ifdef __cplusplus
}
#endif

#endif /* _UTILITIES_SECXPCUTILS_H_ */
