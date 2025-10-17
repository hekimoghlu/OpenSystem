/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 3, 2024.
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
 * TrustKeychains.h - manages the standard keychains searched for trusted certificates. 
 */

#ifndef	_TRUST_KEYCHAINS_H_
#define _TRUST_KEYCHAINS_H_

#include <security_utilities/threading.h>
#include <Security/cssmtype.h>
/*
#if defined(__cplusplus)
extern "C" {
#endif	
*/

/*!
 @function SecTrustKeychainsGetMutex
 @abstract Get the global mutex for accessing trust keychains during an evaluation
 @return On return, a reference to the global mutex which manages access to trust keychains
 @discussion This function is intended to be used by C++ implementation layers to share a
 common global mutex for managing access to trust keychains (i.e. the root certificate store).
 */
RecursiveMutex& SecTrustKeychainsGetMutex();

/*
#if defined(__cplusplus)
}
#endif
*/

#endif	/* _TRUST_KEYCHAINS_H_ */

