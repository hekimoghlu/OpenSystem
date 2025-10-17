/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 25, 2024.
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
#ifndef _TESTCPP_H_
#define _TESTCPP_H_  1

#include "testmore.h"

#ifdef __cplusplus

#define no_throw(THIS, TESTNAME) \
({ \
    bool _this; \
    try { THIS; _this = true; } catch (...) { _this = false; } \
    test_ok(_this, TESTNAME, test_directive, test_reason, \
		__FILE__, __LINE__, \
		"#          got: <unknown exception>\n" \
		"#     expected: <no throw>\n"); \
})
#define does_throw(THIS, TESTNAME) \
({ \
    bool _this; \
    try { THIS; _this = false; } catch (...) { _this = true; } \
    test_ok(_this, TESTNAME, test_directive, test_reason, \
		__FILE__, __LINE__, \
		"#          got: <no throw>\n" \
		"#     expected: <any exception>\n"); \
})
#define is_throw(THIS, CLASS, METHOD, VALUE, TESTNAME) \
({ \
    bool _this; \
    try \
	{ \
		THIS; \
		_this = test_ok(false, TESTNAME, test_directive, test_reason, \
			__FILE__, __LINE__, \
			"#          got: <no throw>\n" \
			"#     expected: %s.%s == %s\n", \
			#CLASS, #METHOD, #VALUE); \
	} \
    catch (const CLASS &_exception) \
    { \
		_this = test_ok(_exception.METHOD == (VALUE), TESTNAME, \
			test_directive, test_reason, __FILE__, __LINE__, \
			"#          got: %d\n" \
			"#     expected: %s.%s == %s\n", \
			_exception.METHOD, #CLASS, #METHOD, #VALUE); \
	} \
    catch (...) \
    { \
    	_this = test_ok(false, TESTNAME, test_directive, test_reason, \
			__FILE__, __LINE__, \
			"#          got: <unknown exception>\n" \
			"#     expected: %s.%s == %s\n", \
			#CLASS, #METHOD, #VALUE); \
	} \
	_this; \
})
#endif /* __cplusplus */

#endif /* !_TESTCPP_H_ */
