/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 12, 2023.
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

#ifndef __FAIL_H_
#define __FAIL_H_

#define TOSTRING_HELPER(x) #x
#define TOSTRING(x) TOSTRING_HELPER(x)

#define PERFINDEX_FAILURE -1
#define PERFINDEX_SUCCESS 0

extern char* error_str;

#define FAIL(message) do {\
    error_str = message " at " __FILE__ ": " TOSTRING(__LINE__);\
    return PERFINDEX_FAILURE;\
} while(0)

#define VERIFY(condition, fail_message) do {\
    if(!(condition)) FAIL(fail_message);\
} while(0)

#endif
