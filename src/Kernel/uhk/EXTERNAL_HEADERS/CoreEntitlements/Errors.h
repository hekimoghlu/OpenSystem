/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 5, 2025.
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
//  Errors.h
//  CoreEntitlements
//
//

#ifndef CORE_ENTITLEMENTS_ERRORS_H
#define CORE_ENTITLEMENTS_ERRORS_H

#ifndef _CE_INDIRECT
#error "Please include <CoreEntitlements/CoreEntitlements.h> instead of this file"
#endif

#include <sys/cdefs.h>
__ptrcheck_abi_assume_single();

/*!
 * @typedef CEError_t
 * A shared error type that is returned by CoreEntitlements APIs
 */
typedef const struct CEError *__single CEError_t;

// Macro to define the error for export;
#define CE_DEF_ERROR(name) extern CEError_t name;

// Returned on successful invocation
CE_DEF_ERROR(kCENoError);

// Returned when the library encounters API misuse
CE_DEF_ERROR(kCEAPIMisuse);

// Returned when an invalid argument has been passed in
CE_DEF_ERROR(kCEInvalidArgument);

// Returned when we expected to have allocated data, but we couldn't
CE_DEF_ERROR(kCEAllocationFailed);

// Returned when the passed in entitlements do not conform to any supported format
CE_DEF_ERROR(kCEMalformedEntitlements);

// Returned when a group of queries does not generate a valid result on the current CEQueryContext
CE_DEF_ERROR(kCEQueryCannotBeSatisfied);

// Returned when a context shouldn't be accelerated
CE_DEF_ERROR(kCENotEligibleForAcceleration);

/*!
 * @function CEGetErrorString
 * Returns a string that describes the error
 */
const char *__unsafe_indexable CEGetErrorString(CEError_t error);

#endif
