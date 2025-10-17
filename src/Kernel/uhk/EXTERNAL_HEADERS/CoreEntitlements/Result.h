/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 8, 2025.
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
//  Result.h
//  CoreEntitlements
//

#ifndef CORE_ENTITLEMENTS_RESULT_H
#define CORE_ENTITLEMENTS_RESULT_H

#ifndef _CE_INDIRECT
#error "Please include <CoreEntitlements/CoreEntitlements.h> instead of this file"
#endif

#include <sys/cdefs.h>
__ptrcheck_abi_assume_single();

#include <CoreEntitlements/Errors.h>
#include <stdint.h>

/*!
 * @function CEErrorPassThrough
 * Returns its argument. Convenient breakpoint location for when anything raises an error.
 */
static inline CEError_t CEErrorPassThrough(CEError_t E) {
    return E;
}

/*!
 * @function CE_CHECK
 * Checks if the passed in return value from one of CoreEntitlements function is an error, and if so returns that error in the current function
 */
#define CE_CHECK(ret) do { CEError_t _ce_error = ret; if (_ce_error != kCENoError) {return CEErrorPassThrough(_ce_error);} } while(0)

/*!
 * @function CE_THROW
 * Macro to "throw" (return) one of the CEErrors
 */
#define CE_THROW(err) return CEErrorPassThrough(err)

/*!
 * @function CE_OK
 * Returns a true if the passed in value corresponds to kCENoError
 */
#define CE_OK(ret) ((ret) == kCENoError)

#endif
