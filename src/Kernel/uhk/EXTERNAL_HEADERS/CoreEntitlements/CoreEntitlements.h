/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 19, 2022.
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
#ifndef CORE_ENTITLEMENTS_H
#define CORE_ENTITLEMENTS_H

#ifdef __cplusplus
extern "C" {
#endif

typedef const struct CERuntime* CERuntime_t;
typedef struct CEQueryContext* CEQueryContext_t;

#define _CE_INDIRECT 1

#ifdef TXM
#include <attributes.h>
#include <ptrcheck.h>
#endif

#if defined(__has_feature) && __has_feature(bounds_attributes)
#define CE_HEADER_INDEXABLE __attribute__((__indexable__))
#else
#define CE_HEADER_INDEXABLE
#endif

#include <os/base.h>
#include <CoreEntitlements/Errors.h>
#include <CoreEntitlements/Result.h>
#include <CoreEntitlements/Runtime.h>
#include <CoreEntitlements/Entitlements.h>
#include <CoreEntitlements/Serialization.h>
#include <CoreEntitlements/Index.h>

__ptrcheck_abi_assume_single();

/*!
 * @typedef CEType_t
 * @brief Represents a type of element supported by CoreEntitlements
 *
 * @const kCETypeUnknown
 * An unknown type
 *
 * @const kCETypeDictionary
 * A dictionary container
 *
 * @const kCETypeSequence
 * An ordered sequence container
 *
 * @const kCETypeInteger
 * An integer.
 *
 * @const kCETypeString
 * A string of bytes.
 *
 * @const kCETypeBool
 * A boolean.
 */
OS_CLOSED_ENUM(CEType, uint32_t,
               kCETypeUnknown = 0,
               kCETypeDictionary = 1,
               kCETypeSequence = 2,
               kCETypeInteger = 3,
               kCETypeString = 4,
               kCETypeBool = 5,
               kCETypeData = 6);

/*!
 * @function CE_RT_LOG
 * Log a single message via the current runtime
 * Only called if the runtime supports logging.
 */
#define CE_RT_LOG(msg) do { if (rt->log) { rt->log(rt, "[%s]: %s\n", __FUNCTION__, msg); } } while(0)

/*!
 * @function CE_RT_LOGF
 * Logs using the passed in format. Printf like.
 * Only called if the runtime supports logging.
 */
#define CE_RT_LOGF(fmt, ...) do { if (rt->log) { rt->log(rt, "[%s]: " fmt, __FUNCTION__, __VA_ARGS__); } } while(0)

/*!
 * @function CE_RT_ABORT
 * Invokes the runtime abort function with a passed in message.
 * This function should not return.
 */
#define CE_RT_ABORT(...) do { rt->abort(rt, "[%s]: %s\n", __FUNCTION__, __VA_ARGS__); } while(0)

#ifdef __cplusplus
}
#endif

#endif
