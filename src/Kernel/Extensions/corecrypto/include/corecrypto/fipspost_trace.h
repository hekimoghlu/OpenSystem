/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 1, 2025.
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
#ifndef _CORECRYPTO_FIPSPOST_TRACE_H_
#define _CORECRYPTO_FIPSPOST_TRACE_H_

#if CC_FIPSPOST_TRACE

/*
 * Use this string to separate out tests.
 */
#define FIPSPOST_TRACE_TEST_STR    "?"

int fipspost_trace_is_active(void);
void fipspost_trace_call(const char *fname);

/* Only trace when VERBOSE is set to avoid impacting normal boots. */
#define FIPSPOST_TRACE_EVENT do {                                       \
    if (fipspost_trace_is_active()) {                                   \
        fipspost_trace_call(__FUNCTION__);                              \
    }                                                                   \
} while (0);

#define FIPSPOST_TRACE_MESSAGE(MSG) do {                                \
    if (fipspost_trace_is_active()) {                                   \
        fipspost_trace_call(MSG);                                       \
    }                                                                   \
} while (0);

#else

/* Not building a CC_FIPSPOST_TRACE-enabled, no TRACE operations. */
#define FIPSPOST_TRACE_EVENT
#define FIPSPOST_TRACE_MESSAGE(X)

#endif

#endif
