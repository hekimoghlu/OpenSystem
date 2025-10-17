/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 11, 2024.
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
#ifndef JSBasePrivate_h
#define JSBasePrivate_h

#include <JavaScriptCore/JSBase.h>
#include <JavaScriptCore/WebKitAvailability.h>

#ifdef __cplusplus
extern "C" {
#endif

/*!
@function
@abstract Reports an object's non-GC memory payload to the garbage collector.
@param ctx The execution context to use.
@param size The payload's size, in bytes.
@discussion Use this function to notify the garbage collector that a GC object
owns a large non-GC memory region. Calling this function will encourage the
garbage collector to collect soon, hoping to reclaim that large non-GC memory
region.
*/
JS_EXPORT void JSReportExtraMemoryCost(JSContextRef ctx, size_t size) JSC_API_AVAILABLE(macos(10.6), ios(7.0));

JS_EXPORT void JSDisableGCTimer(void);

#if !defined(__APPLE__) && !defined(WIN32) && !defined(_WIN32)
/*!
@function JSConfigureSignalForGC
@abstract Configure signals for GC in non-Apple and non-Windows platforms.
@param signal The signal number to use.
@result true if the signal is successfully configured, otherwise false.
@discussion Call this function before any of JSC initialization starts. Otherwise, it fails.
*/
JS_EXPORT bool JSConfigureSignalForGC(int signal);
#endif

/*!
@function
@abstract Produces an object with various statistics about current memory usage.
@param ctx The execution context to use.
@result An object containing GC heap status data.
@discussion Specifically, the result object has the following integer-valued fields:
 heapSize: current size of heap
 heapCapacity: current capacity of heap
 extraMemorySize: amount of non-GC memory referenced by GC objects (included in heap size / capacity)
 objectCount: current count of GC objects
 protectedObjectCount: current count of protected GC objects
 globalObjectCount: current count of global GC objects
 protectedGlobalObjectCount: current count of protected global GC objects
 objectTypeCounts: object with GC object types as keys and their current counts as values
*/
JS_EXPORT JSObjectRef JSGetMemoryUsageStatistics(JSContextRef ctx);

#ifdef __cplusplus
}
#endif

#endif /* JSBasePrivate_h */
