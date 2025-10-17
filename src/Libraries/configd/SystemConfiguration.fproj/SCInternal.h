/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 18, 2024.
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
#ifndef _SCINTERNAL_H
#define _SCINTERNAL_H

/*
 * Internal status codes
 * - handled on the framework (client) side and mapped to their respective
 *   regular kSCStatus{AccessError,OK} codes before returning
 */
enum {
	kSCStatusAccessError_MissingAuthorization	= 10001,
	kSCStatusAccessError_MissingWriteEntitlement	= 10002,
	kSCStatusAccessError_MissingReadEntitlement	= 10003,
	kSCStatusOK_MissingReadEntitlement		= 10100,
};

#endif /* _SCINTERNAL_H */
