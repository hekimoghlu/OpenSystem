/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 21, 2021.
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
// CSSM-style C/C++ bridge facilities
//
#ifndef _H_CSSMBRIDGE
#define _H_CSSMBRIDGE

#include <security_utilities/utilities.h>
#include <security_cdsa_utilities/cssmerrors.h>
#include <Security/cssm.h>

namespace Security {

//
// API boilerplate macros. These provide a frame for C++ code that is impermeable to exceptions.
// Usage:
//	BEGIN_API
//		... your C++ code here ...
//  END_API(base)	// returns CSSM_RETURN on exception; complete it to 'base' (DL, etc.) class;
//					// returns CSSM_OK on fall-through
//	END_API0		// completely ignores exceptions; falls through in all cases
//	END_API1(bad)	// return (bad) on exception; fall through on success
//
#define BEGIN_API \
	CSSM_RETURN __attribute__((unused)) __retval = CSSM_OK;	  \
	bool __countlegacyapi __attribute__((cleanup(setCountLegacyAPIEnabledForThreadCleanup))) = countLegacyAPIEnabledForThread(); \
	static dispatch_once_t countToken; \
	countLegacyAPI(&countToken, __FUNCTION__); \
	setCountLegacyAPIEnabledForThread(false); \
	try {

#define BEGIN_API_NO_METRICS \
	CSSM_RETURN __attribute__((unused)) __retval = CSSM_OK;	\
	try {

#define END_API(base)	} \
	catch (const CommonError &err) { __retval = CssmError::cssmError(err, CSSM_ ## base ## _BASE_ERROR); } \
	catch (const std::bad_alloc &) { __retval = CssmError::cssmError(CSSM_ERRCODE_MEMORY_ERROR, CSSM_ ## base ## _BASE_ERROR); } \
	catch (...) { __retval = CssmError::cssmError(CSSM_ERRCODE_INTERNAL_ERROR, CSSM_ ## base ## _BASE_ERROR); } \
	return __retval;
#define END_API0		} \
	catch (...) {} \
	return;
#define END_API1(bad)	} \
	catch (...) { __retval = bad; } \
	return __retval;

#define END_API_NO_METRICS(base)	} \
	catch (const CommonError &err) { __retval = CssmError::cssmError(err, CSSM_ ## base ## _BASE_ERROR); } \
	catch (const std::bad_alloc &) { __retval = CssmError::cssmError(CSSM_ERRCODE_MEMORY_ERROR, CSSM_ ## base ## _BASE_ERROR); } \
	catch (...) { __retval = CssmError::cssmError(CSSM_ERRCODE_INTERNAL_ERROR, CSSM_ ## base ## _BASE_ERROR); } \
	return __retval;
#define END_API0_NO_METRICS		} \
	catch (...) {} \
	return;
#define END_API1_NO_METRICS(bad)	} \
	catch (...) {} \
	return bad;

} // end namespace Security


#endif //_H_CSSMBRIDGE
