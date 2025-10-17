/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 15, 2025.
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
#ifndef __CrashReporterClient__
#define __CrashReporterClient__

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#define CRASHREPORTER_ANNOTATIONS_SECTION "__crash_info"
#define CRASHREPORTER_ANNOTATIONS_VERSION 5
#define CRASH_REPORTER_CLIENT_HIDDEN __attribute__((visibility("hidden")))

#define _crc_make_getter(attr) ((const char *)gCRAnnotations.attr)
#define _crc_make_setter(attr, arg) (gCRAnnotations.attr = (uint64_t)(arg))
#define CRGetCrashLogMessage() _crc_make_getter(message)
#define CRSetCrashLogMessage(m) _crc_make_setter(message, m)
#define CRGetCrashLogMessage2() _crc_make_getter(message2)
#define CRSetCrashLogMessage2(m) _crc_make_setter(message2, m)

struct crashreporter_annotations_t {
	uint64_t version;
	uint64_t message;
	uint64_t signature_string;
	uint64_t backtrace;
	uint64_t message2;
	uint64_t thread;
	uint64_t dialog_mode;
	uint64_t abort_cause;
};

CRASH_REPORTER_CLIENT_HIDDEN
extern struct crashreporter_annotations_t gCRAnnotations;

#ifdef __cplusplus
};
#endif

#endif /* __CrashReporterClient__ */
