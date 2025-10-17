/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 24, 2023.
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
 *  KCExceptions.h
 */
#ifndef _SECURITY_KCEXCEPTIONS_H_
#define _SECURITY_KCEXCEPTIONS_H_

#include <security_utilities/errors.h>
#include <Security/SecBase.h>
#ifdef lock
#undef lock
#endif
//#include <security_cdsa_utilities/utilities.h>

#ifdef check
#undef check
#endif

namespace Security
{

namespace KeychainCore
{

//
// Helpers for memory pointer validation
//

/*	remove RequiredParam when cdsa does namespaces
template <class T>
inline T &Required(T *ptr,OSStatus err = errSecParam)
{
    return Required(ptr,err);
}
*/

template <class T>
inline void KCThrowIfMemFail_(const T *ptr)
{
    if (ptr==NULL)
		MacOSError::throwMe(errSecAllocate);
}

inline void KCThrowIf_(OSStatus theErr)
{
	// will also work for OSErr
    if (theErr!=errSecSuccess)
        MacOSError::throwMe(theErr);
}

inline void KCThrowIf_(bool test,OSStatus theErr)
{
	// will also work for OSErr
    if (test)
        MacOSError::throwMe(theErr);
}

inline void KCThrowParamErrIf_(bool test)
{
    if (test)
        MacOSError::throwMe(errSecParam);
}

inline void KCUnimplemented_()
{
	MacOSError::throwMe(errSecUnimplemented);
}

} // end namespace KeychainCore

} // end namespace Security

#endif /* !_SECURITY_KCEXCEPTIONS_H_ */
