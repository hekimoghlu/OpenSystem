/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 3, 2023.
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
 * cssmerrors
 */
#ifndef _H_CSSMERRORS
#define _H_CSSMERRORS

#include <security_utilities/errors.h>
#include <Security/cssmtype.h>

namespace Security
{

//
// A CSSM-originated error condition, represented by a CSSM_RETURN value.
// This can represent both a convertible base error, or a module-specific
// error condition.
//
class CssmError : public CommonError {
protected:
    CssmError(CSSM_RETURN err, bool suppresslogging);
public:
    const CSSM_RETURN error;
    virtual OSStatus osStatus() const;
	virtual int unixError() const;
    virtual const char *what () const _NOEXCEPT;

    static CSSM_RETURN merge(CSSM_RETURN error, CSSM_RETURN base);
    
	static void check(CSSM_RETURN error)	{ if (error != CSSM_OK) throwMe(error); }
    static void throwMe(CSSM_RETURN error) __attribute__((noreturn));
    static void throwMeNoLogging(CSSM_RETURN err) __attribute__((noreturn));

	//
	// Obtain a CSSM_RETURN from any CommonError
	//
	static CSSM_RETURN cssmError(CSSM_RETURN error, CSSM_RETURN base);
	static CSSM_RETURN cssmError(const CommonError &error, CSSM_RETURN base);
};



} // end namespace Security


#endif //_H_CSSMERRORS
