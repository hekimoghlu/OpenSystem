/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 27, 2023.
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
 * YarrowConnection.h - single, process-wide, thread-safe Yarrow client
 */

#ifndef	_YARROW_CONNECTION_H_
#define _YARROW_CONNECTION_H_

#ifdef	__cplusplus
extern "C" {
#endif

/*
 * Both functions a CssmError::throwMe(CSSMERR_CSP_FUNCTION_FAILED) on failure. 
 * 
 * "Give me some random data". Caller mallocs the data. 
 */
extern void	cspGetRandomBytes(void *buf, unsigned len);

/*
 * Add some entropy to the pool. 
 */
extern void cspAddEntropy(const void *buf, unsigned len);

#ifdef	__cplusplus
}
#endif

#endif	/* _YARROW_CONNECTION_H_ */
